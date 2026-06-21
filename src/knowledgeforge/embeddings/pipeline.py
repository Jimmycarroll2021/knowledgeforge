"""Embedding pipeline — Phase 4 of the KnowledgeForge roadmap.

Generates dense vector embeddings for every entity in the graph by combining
two signals:

  - Semantic: a sentence-transformer encodes each entity's text (its id plus
    its top evidence triples) into a 384-dim base vector.
  - Structural: a *learned* GraphSAGE MEAN aggregator (Hamilton, Ying &
    Leskovec, 2017 — "Inductive Representation Learning on Large Graphs",
    Algorithm 1) folds in neighbour information.

The aggregator is a genuine learned layer, not a fixed average. For each node

    h_v = L2normalize( ReLU( W · CONCAT(h_v_self, MEAN({h_u : u ∈ N(v)})) ) )

with a single weight matrix ``W`` of shape ``(out_dim, 2*in_dim)`` (here
in_dim == out_dim == 384). ``W`` is trained UNSUPERVISED with GraphSAGE's
graph-based loss (no labels exist in a knowledge graph):

    J_G(z_u) = -log σ(z_u · z_v) - Q · E_{v_n ~ P_n}[ log σ(-z_u · z_{v_n}) ]

Positive pairs (u, v) are drawn from short random walks over the semantic
neighbour graph; ``Q`` negatives are sampled uniformly per positive. Forward
pass and back-propagation through the single Linear→ReLU→L2 layer are written
by hand in numpy and optimised with plain SGD. The aggregator is applied twice
(two layers sharing ``W``) so structural signal propagates up to K=2 hops.
The trained ``W`` is persisted to ``{chroma_path}/graphsage_w.npy`` and reused
on later builds for reproducibility.

Fast similarity search via turbovec (Google TurboQuant — 4-bit SIMD quantisation).
Full metadata store via ChromaDB (persistent, filterable by entity kind).
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..store.sqlite import SQLiteGraphStore

_DEFAULT_MODEL = "all-MiniLM-L6-v2"  # 384-dim, fast, good quality
_EMBED_DIM = 384
_SEED = 42  # all randomness seeded from here for determinism

# Predicates that carry structural noise rather than semantic relatedness —
# excluded from the neighbour graph used for aggregation and random walks.
_NON_SEMANTIC_PREDICATES = ("HAS_FILE_TYPE", "CONTAINS_KEY", "HAS_TAG")


def _sigmoid(x: float) -> float:
    """Numerically stable scalar logistic sigmoid."""
    if x >= 0:
        return float(1.0 / (1.0 + np.exp(-x)))
    e = np.exp(x)
    return float(e / (1.0 + e))


class EmbeddingPipeline:
    """Builds and queries entity embeddings for the knowledge graph.

    Embeddings = sentence-transformer base vectors refined by a *learned*
    GraphSAGE MEAN aggregator (see module docstring). The aggregator weight
    ``W`` is trained unsupervised on the graph and cached to disk.

    Usage:
        pipeline = EmbeddingPipeline(store, chroma_path="data/embeddings")
        pipeline.embed_all()                          # build embeddings
        results = pipeline.find_similar("GraphSAGE")  # query
    """

    # GraphSAGE hyperparameters (small, fast, deterministic).
    _SAGE_LAYERS = 2          # depth K — number of aggregator hops
    _SAGE_EPOCHS = 300        # SGD steps
    _SAGE_LR = 0.01           # learning rate
    _SAGE_NEG_SAMPLES = 10    # Q — negatives per positive pair
    _WALKS_PER_NODE = 50      # random walks per node (scaled down on small graphs)
    _WALK_LENGTH = 5          # length of each random walk
    _WALK_BUDGET = 200        # total starting-walk budget across active nodes

    def __init__(
        self,
        store: "SQLiteGraphStore",
        chroma_path: str | Path = "data/embeddings",
        model_name: str = _DEFAULT_MODEL,
    ) -> None:
        self._store = store
        self._chroma_path = str(Path(chroma_path).resolve())
        self._model_name = model_name
        self._model: Any = None
        self._chroma: Any = None
        self._collection: Any = None
        self._turbo: Any = None
        self._turbo_ids: list[str] = []
        self._turbo_pad: int = 0
        self._last_train_losses: list[float] = []
        self._w_path = Path(self._chroma_path) / "graphsage_w.npy"

    # ── public ────────────────────────────────────────────────────────────────

    def embed_all(self, batch_size: int = 64) -> dict[str, Any]:
        """Embed every entity in the graph.

        Two-phase build so the learned GraphSAGE aggregator actually injects
        structure on a cold build:

          1. Encode the base semantic vector for ALL to-embed entities into an
             in-memory ``{entity_id: base_vec}`` dict — neighbours' base vectors
             are therefore available without reading mid-loop from ChromaDB.
          2. Build the semantic neighbour adjacency, train ``W`` on it, then
             apply the aggregator over the in-memory dict (K hops).
          3. Write the aggregated vectors to ChromaDB and rebuild the turbo index.

        Returns stats: {entities_embedded, skipped, model}.
        """
        model = self._get_model()
        collection = self._get_collection()

        entities = self._store._conn.execute(
            "SELECT id, kind FROM entities"
        ).fetchall()

        already = set(collection.get(include=[])["ids"])
        to_embed = [(eid, kind) for eid, kind in entities if eid not in already]

        if not to_embed:
            return {"entities_embedded": 0, "skipped": len(already), "model": self._model_name}

        # ── Phase 1: encode base vectors for every to-embed entity ──────────────
        base_vecs: dict[str, np.ndarray] = {}
        kinds: dict[str, str] = {}
        ids = [eid for eid, _ in to_embed]
        kinds = {eid: kind for eid, kind in to_embed}

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_texts = [self._entity_text(eid, kinds[eid]) for eid in batch_ids]
            vecs = np.asarray(
                model.encode(batch_texts, normalize_embeddings=True),
                dtype=np.float32,
            )
            for eid, vec in zip(batch_ids, vecs):
                base_vecs[eid] = vec

        # ── Phase 2: build adjacency, train aggregator, apply over in-memory dict ─
        adjacency = self._build_adjacency(base_vecs.keys())
        W = self._train_aggregator(base_vecs, adjacency)
        agg_vecs = self._apply_aggregator(base_vecs, adjacency, W)

        # ── Phase 3: write aggregated vectors to ChromaDB ───────────────────────
        embedded = 0
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            collection.add(
                embeddings=[agg_vecs[eid].tolist() for eid in batch_ids],
                ids=batch_ids,
                metadatas=[{"kind": kinds[eid], "entity_id": eid} for eid in batch_ids],
            )
            embedded += len(batch_ids)

        # rebuild turbovec index
        self._build_turbo_index()

        return {
            "entities_embedded": embedded,
            "skipped": len(already),
            "model": self._model_name,
        }

    def find_similar(
        self,
        entity_id: str,
        k: int = 10,
        kind_filter: str | None = None,
        use_turbo: bool = True,
    ) -> list[dict[str, Any]]:
        """Find k most similar entities to entity_id.

        Returns: [{entity_id, kind, score}, ...]

        use_turbo=True uses turbovec (fast 4-bit SIMD).
        use_turbo=False falls back to ChromaDB cosine query.
        """
        if use_turbo and self._turbo is not None:
            return self._turbo_search(entity_id, k, kind_filter)
        return self._chroma_search(entity_id, k, kind_filter)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a raw text query for similarity search."""
        model = self._get_model()
        vec = np.asarray(model.encode([text], normalize_embeddings=True))
        return vec[0]  # type: ignore[no-any-return]  # ndarray.__getitem__ is typed Any

    def search_by_text(
        self, query: str, k: int = 10, kind_filter: str | None = None
    ) -> list[dict[str, Any]]:
        """Find entities most similar to a free-text query."""
        collection = self._get_collection()
        where = {"kind": kind_filter} if kind_filter else None
        results = collection.query(
            query_texts=[query],
            n_results=k,
            where=where,
            include=["metadatas", "distances"],
        )
        out: list[dict[str, Any]] = []
        for eid, meta, dist in zip(
            results["ids"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            out.append({"entity_id": eid, "kind": meta.get("kind"), "score": 1 - dist})
        return out

    def stats(self) -> dict[str, Any]:
        col = self._get_collection()
        count = col.count()
        return {
            "embedded_entities": count,
            "model": self._model_name,
            "chroma_path": self._chroma_path,
            "turbo_index_size": len(self._turbo_ids) if self._turbo_ids else 0,
        }

    # ── private ───────────────────────────────────────────────────────────────

    def _entity_text(self, entity_id: str, kind: str) -> str:
        """Build a text representation for an entity using its triples as context."""
        rows = self._store._conn.execute(
            """SELECT predicate, object, evidence
               FROM triples
               WHERE subject = ?
               ORDER BY confidence DESC
               LIMIT 8""",
            (entity_id,),
        ).fetchall()

        parts = [entity_id]
        for pred, obj, evidence in rows:
            if pred not in ("HAS_FILE_TYPE", "CONTAINS_KEY"):
                parts.append(f"{pred}: {obj}")
                if evidence and len(evidence) > 10:
                    parts.append(evidence[:120])
        return " | ".join(parts)

    # ── learned GraphSAGE aggregator (Hamilton 2017, Algorithm 1) ───────────────

    def _build_adjacency(self, entity_ids: Iterable[str]) -> dict[str, list[str]]:
        """Undirected semantic-neighbour adjacency from the triples table.

        Edges from non-semantic predicates (HAS_FILE_TYPE / CONTAINS_KEY /
        HAS_TAG) are excluded. Only edges between entities present in
        ``entity_ids`` are kept (so neighbour vectors are always available).
        """
        present = set(entity_ids)
        placeholders = ",".join("?" * len(_NON_SEMANTIC_PREDICATES))
        rows = self._store._conn.execute(
            f"""SELECT DISTINCT subject, object FROM triples
                WHERE predicate NOT IN ({placeholders})""",
            _NON_SEMANTIC_PREDICATES,
        ).fetchall()

        adjacency: dict[str, list[str]] = {eid: [] for eid in present}
        seen: set[tuple[str, str]] = set()
        for subj, obj in rows:
            if subj == obj or subj not in present or obj not in present:
                continue
            key = (subj, obj) if subj < obj else (obj, subj)
            if key in seen:
                continue
            seen.add(key)
            adjacency[subj].append(obj)
            adjacency[obj].append(subj)
        return adjacency

    def _random_walks(
        self, adjacency: dict[str, list[str]], rng: np.random.Generator
    ) -> list[tuple[str, str]]:
        """Generate positive co-occurrence pairs from short random walks.

        Walks of length ``_WALK_LENGTH`` are started from every non-isolated
        node (``_WALKS_PER_NODE`` walks each, scaled down for small graphs).
        Every consecutive (current, next) step yields a positive pair.
        """
        nodes = [n for n, nbrs in adjacency.items() if nbrs]
        if not nodes:
            return []

        # Per-epoch scatter cost grows with the pair count, so bound the total
        # walk budget: ~_WALK_BUDGET starting walks split across the active
        # nodes (at least one walk per node).
        walks_per_node = max(1, min(self._WALKS_PER_NODE, self._WALK_BUDGET // len(nodes)))

        pairs: list[tuple[str, str]] = []
        for start in nodes:
            for _ in range(walks_per_node):
                cur = start
                for _ in range(self._WALK_LENGTH):
                    nbrs = adjacency[cur]
                    if not nbrs:
                        break
                    nxt = nbrs[rng.integers(len(nbrs))]
                    pairs.append((cur, nxt))
                    cur = nxt
        return pairs

    def _forward(
        self, self_vec: np.ndarray, neigh_mean: np.ndarray, W: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One aggregator layer. Returns (z, pre_relu, concat) for backprop.

        z = L2normalize( ReLU( W · CONCAT(self_vec, neigh_mean) ) )
        """
        concat = np.concatenate([self_vec, neigh_mean])
        pre = W @ concat                       # (out_dim,)
        relu = np.maximum(pre, 0.0)
        norm = np.linalg.norm(relu)
        z = relu / norm if norm > 0 else relu
        return z, pre, concat

    def _neighbour_means(
        self, vecs: dict[str, np.ndarray], adjacency: dict[str, list[str]]
    ) -> dict[str, np.ndarray]:
        """Mean of each node's neighbour vectors (zeros if isolated)."""
        dim = next(iter(vecs.values())).shape[0]
        means: dict[str, np.ndarray] = {}
        for eid, vec in vecs.items():
            nbrs = adjacency.get(eid, [])
            if nbrs:
                means[eid] = np.mean([vecs[n] for n in nbrs], axis=0)
            else:
                means[eid] = np.zeros(dim, dtype=np.float32)
        return means

    def _train_aggregator(
        self, base_vecs: dict[str, np.ndarray], adjacency: dict[str, list[str]]
    ) -> np.ndarray:
        """Train the GraphSAGE weight matrix W unsupervised.

        Minimises the graph-based loss

            J(z_u) = -log σ(z_u·z_v) - Q·E_{v_n~P_n}[ log σ(-z_u·z_{v_n}) ]

        over positive pairs from random walks, with Q uniform negatives each.
        Forward + manual backprop through a single Linear→ReLU→L2 layer, SGD.
        Deterministic: all randomness comes from ``np.random.default_rng(_SEED)``.

        Returns W of shape ``(out_dim, 2*in_dim)``. Records per-step loss in
        ``self._last_train_losses``. If a cached W exists at ``_w_path`` it is
        loaded and returned without retraining.
        """
        in_dim = next(iter(base_vecs.values())).shape[0] if base_vecs else _EMBED_DIM
        out_dim = in_dim

        if self._w_path.exists():
            cached: np.ndarray = np.load(self._w_path)
            if cached.shape == (out_dim, 2 * in_dim):
                self._last_train_losses = []
                return cached

        rng = np.random.default_rng(_SEED)

        # Identity-biased init: the self-block starts near identity so an
        # untrained aggregator roughly preserves the base vector, and small
        # random weights on both blocks give gradient signal to learn from.
        W = rng.normal(0.0, 0.01, size=(out_dim, 2 * in_dim)).astype(np.float32)
        W[:, :in_dim] += np.eye(out_dim, in_dim, dtype=np.float32)

        pairs = self._random_walks(adjacency, rng)
        neigh_mean = self._neighbour_means(base_vecs, adjacency)
        node_ids = list(base_vecs.keys())
        n_nodes = len(node_ids)
        idx_of = {eid: i for i, eid in enumerate(node_ids)}

        losses: list[float] = []
        if not pairs or n_nodes < 2:
            # Nothing to learn from — keep init, report a flat (zero-step) loss.
            self._last_train_losses = losses
            np.save(self._w_path, W)
            return W

        q = self._SAGE_NEG_SAMPLES
        lr = self._SAGE_LR

        # The aggregator input CONCAT(self, mean(neighbours)) is fixed across
        # training (base vectors and adjacency don't change), so build it once.
        # Each epoch is then a handful of batched matmuls rather than a Python
        # loop over pairs — keeps a few-hundred-node graph well under 30s.
        concat = np.empty((n_nodes, 2 * in_dim), dtype=np.float32)
        for eid, i in idx_of.items():
            concat[i, :in_dim] = base_vecs[eid]
            concat[i, in_dim:] = neigh_mean[eid]

        # Collapse the (often heavily repeated) walk pairs into unique directed
        # edges with integer multiplicities. The per-epoch cost then scales with
        # the number of *distinct* co-occurring pairs, not the raw walk count —
        # so a tiny graph trains in well under a second while large graphs stay
        # bounded. Multiplicity acts as the empirical co-occurrence weight that
        # GraphSAGE's expected-co-occurrence loss is defined over.
        raw = np.array([(idx_of[u], idx_of[v]) for u, v in pairs], dtype=np.int64)
        uniq, counts = np.unique(raw, axis=0, return_counts=True)
        ui = uniq[:, 0]
        vi = uniq[:, 1]
        w = counts.astype(np.float64)                   # pair weights
        w_col = w[:, None]
        total_w = float(w.sum())
        n_pairs = len(ui)

        for _ in range(self._SAGE_EPOCHS):
            # Forward for every node: Z = L2norm(ReLU(concat @ Wᵀ)).
            pre = concat @ W.T                          # (N, out)
            relu = np.maximum(pre, 0.0)
            norms = np.linalg.norm(relu, axis=1, keepdims=True)
            safe = np.where(norms > 0, norms, 1.0)
            Z = relu / safe                             # (N, out)

            total_loss = 0.0
            # Per-pair gradient w.r.t. Z is always a scalar coefficient times the
            # partner's row. Rather than scatter full (dim,) vectors (slow), we
            # accumulate scalar coefficients into a node×node matrix C and recover
            # grad_Z with two dense matmuls — C @ Z and Cᵀ @ Z. This keeps the
            # per-epoch cost dominated by cheap matmuls even on larger graphs.
            C = np.zeros((n_nodes, n_nodes), dtype=np.float64)

            # Positive term: -log σ(z_u · z_v), weighted by co-occurrence count.
            dot_uv = np.einsum("ij,ij->i", Z[ui], Z[vi])
            puv = 1.0 / (1.0 + np.exp(-dot_uv))
            total_loss += float((w * -np.log(puv + 1e-12)).sum())
            pos_coef = -(1.0 - puv) * w                 # weighted dL/d(dot)
            np.add.at(C, (ui, vi), pos_coef)

            # Negative term: -Q·E[ log σ(-z_u · z_n) ], Q uniform negatives per pair.
            neg = rng.integers(0, n_nodes, size=(n_pairs, q))   # (P, Q)
            dot_un = np.einsum("ij,ikj->ik", Z[ui], Z[neg])     # (P, Q)
            pn = 1.0 / (1.0 + np.exp(dot_un))                   # σ(-dot)
            total_loss += float((w_col * -np.log(pn + 1e-12)).sum())
            neg_coef = (1.0 - pn) * w_col                       # (P, Q)
            np.add.at(C, (np.repeat(ui, q), neg.ravel()), neg_coef.ravel())

            # grad_Z[a] = Σ_b C[a,b]·Z[b]  and  grad_Z[b] += Σ_a C[a,b]·Z[a]
            grad_Z = C @ Z + C.T @ Z

            # Backprop grad_Z through L2-norm then ReLU into grad_W.
            # d(relu/‖relu‖)/d(relu) = (I - z zᵀ)/‖relu‖
            proj = np.einsum("ij,ij->i", grad_Z, Z)[:, None]
            grad_relu = (grad_Z - Z * proj) / safe
            grad_pre = grad_relu * (pre > 0)            # ReLU local grad
            grad_W = grad_pre.T @ concat                # (out, 2*in)

            W = W - lr * (grad_W / total_w)
            losses.append(total_loss / total_w)

        self._last_train_losses = losses
        np.save(self._w_path, W)
        return W

    def _apply_aggregator(
        self,
        base_vecs: dict[str, np.ndarray],
        adjacency: dict[str, list[str]],
        W: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Apply the trained aggregator over the in-memory vectors.

        Runs ``_SAGE_LAYERS`` layers sharing ``W`` so structure propagates up to
        K hops. Isolated nodes (no neighbours) fall back to their L2-normalised
        base vector at every layer.
        """
        cur = base_vecs
        for _ in range(self._SAGE_LAYERS):
            neigh_mean = self._neighbour_means(cur, adjacency)
            nxt: dict[str, np.ndarray] = {}
            for eid, vec in cur.items():
                if adjacency.get(eid):
                    z, _, _ = self._forward(vec, neigh_mean[eid], W)
                    nxt[eid] = z.astype(np.float32)
                else:
                    norm = np.linalg.norm(vec)
                    nxt[eid] = (vec / norm if norm > 0 else vec).astype(np.float32)
            cur = nxt
        return cur

    def _build_turbo_index(self) -> None:
        """Build turbovec SIMD index from ChromaDB embeddings."""
        try:
            from turbovec import TurboQuantIndex
        except ImportError:
            return

        collection = self._get_collection()
        result = collection.get(include=["embeddings"])
        if not result["ids"]:
            return

        self._turbo_ids = result["ids"]
        vecs = np.array(result["embeddings"], dtype=np.float32)

        # turbovec requires dim to be a multiple of 8
        dim = vecs.shape[1]
        pad = (8 - dim % 8) % 8
        if pad:
            vecs = np.pad(vecs, ((0, 0), (0, pad)))
            dim = vecs.shape[1]

        idx = TurboQuantIndex(dim=dim, bit_width=4)
        idx.add(vecs)
        idx.prepare()
        self._turbo = idx
        self._turbo_pad = pad

    def _turbo_search(
        self, entity_id: str, k: int, kind_filter: str | None
    ) -> list[dict[str, Any]]:
        """Query using turbovec SIMD index."""
        if entity_id not in self._turbo_ids:
            return self._chroma_search(entity_id, k, kind_filter)

        collection = self._get_collection()
        result = collection.get(ids=[entity_id], include=["embeddings"])
        if result["embeddings"] is None or len(result["embeddings"]) == 0:
            return []

        vec = np.array(result["embeddings"][0], dtype=np.float32)
        if self._turbo_pad:
            vec = np.pad(vec, (0, self._turbo_pad))

        mask = None
        if kind_filter:
            meta_result = collection.get(include=["metadatas"])
            mask = np.array(
                [m.get("kind") == kind_filter for m in meta_result["metadatas"]],
                dtype=bool,
            )

        scores, idxs = self._turbo.search(vec.reshape(1, -1), k=k + 1, mask=mask)
        scores, idxs = scores[0], idxs[0]

        out: list[dict[str, Any]] = []
        for score, idx in zip(scores, idxs):
            eid = self._turbo_ids[idx]
            if eid == entity_id:
                continue
            meta = collection.get(ids=[eid], include=["metadatas"])
            kind = meta["metadatas"][0].get("kind") if meta["metadatas"] else None
            out.append({"entity_id": eid, "kind": kind, "score": float(score)})
            if len(out) >= k:
                break
        return out

    def _chroma_search(
        self, entity_id: str, k: int, kind_filter: str | None
    ) -> list[dict[str, Any]]:
        """Fallback: ChromaDB cosine similarity search."""
        collection = self._get_collection()
        result = collection.get(ids=[entity_id], include=["embeddings"])
        if result["embeddings"] is None or len(result["embeddings"]) == 0:
            return []

        where = {"kind": kind_filter} if kind_filter else None
        qr = collection.query(
            query_embeddings=[result["embeddings"][0]],
            n_results=k + 1,
            where=where,
            include=["metadatas", "distances"],
        )
        out: list[dict[str, Any]] = []
        for eid, meta, dist in zip(
            qr["ids"][0], qr["metadatas"][0], qr["distances"][0]
        ):
            if eid == entity_id:
                continue
            out.append({"entity_id": eid, "kind": meta.get("kind"), "score": 1 - dist})
            if len(out) >= k:
                break
        return out

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _get_collection(self) -> Any:
        if self._collection is None:
            import chromadb
            self._chroma = chromadb.PersistentClient(path=self._chroma_path)
            self._collection = self._chroma.get_or_create_collection(
                name="entities",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection
