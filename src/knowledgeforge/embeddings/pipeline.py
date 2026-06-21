"""Embedding pipeline — Phase 4 of the KnowledgeForge roadmap.

Generates dense vector embeddings for every entity in the graph using
sentence-transformers (semantic) + GraphSAGE-style mean-pool aggregation
of 1-hop neighbours (structural).

Fast similarity search via turbovec (Google TurboQuant — 4-bit SIMD quantisation).
Full metadata store via ChromaDB (persistent, filterable by entity kind).

From ROADMAP.md Phase 4:
  - Semantic embeddings: sentence-transformer on entity + top evidence text
  - GraphSAGE aggregation: mean-pool 1-hop neighbour embeddings
  - find_similar_entities(entity_id, k) → semantically related graph entities
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..store.sqlite import SQLiteGraphStore

_DEFAULT_MODEL = "all-MiniLM-L6-v2"  # 384-dim, fast, good quality
_EMBED_DIM = 384


class EmbeddingPipeline:
    """Builds and queries entity embeddings for the knowledge graph.

    Usage:
        pipeline = EmbeddingPipeline(store, chroma_path="data/embeddings")
        pipeline.embed_all()                          # build embeddings
        results = pipeline.find_similar("GraphSAGE")  # query
    """

    def __init__(
        self,
        store: "SQLiteGraphStore",
        chroma_path: str | Path = "data/embeddings",
        model_name: str = _DEFAULT_MODEL,
    ) -> None:
        self._store = store
        self._chroma_path = str(Path(chroma_path).resolve())
        self._model_name = model_name
        self._model = None
        self._chroma = None
        self._collection = None
        self._turbo: object | None = None
        self._turbo_ids: list[str] = []

    # ── public ────────────────────────────────────────────────────────────────

    def embed_all(self, batch_size: int = 64) -> dict:
        """Embed every entity in the graph.

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

        texts = []
        ids = []
        metadatas = []
        for eid, kind in to_embed:
            text = self._entity_text(eid, kind)
            texts.append(text)
            ids.append(eid)
            metadatas.append({"kind": kind, "entity_id": eid})

        # batch encode
        embedded = 0
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size]

            vecs = model.encode(batch_texts, normalize_embeddings=True)
            # GraphSAGE-style: mean-pool with 1-hop neighbour embeddings
            vecs = self._graphsage_aggregate(vecs, batch_ids, model)

            collection.add(
                embeddings=vecs.tolist(),
                ids=batch_ids,
                metadatas=batch_meta,
            )
            embedded += len(batch_texts)

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
    ) -> list[dict]:
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
        vec = model.encode([text], normalize_embeddings=True)
        return vec[0]

    def search_by_text(self, query: str, k: int = 10, kind_filter: str | None = None) -> list[dict]:
        """Find entities most similar to a free-text query."""
        collection = self._get_collection()
        where = {"kind": kind_filter} if kind_filter else None
        results = collection.query(
            query_texts=[query],
            n_results=k,
            where=where,
            include=["metadatas", "distances"],
        )
        out = []
        for eid, meta, dist in zip(
            results["ids"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            out.append({"entity_id": eid, "kind": meta.get("kind"), "score": 1 - dist})
        return out

    def stats(self) -> dict:
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

    def _graphsage_aggregate(
        self, vecs: np.ndarray, ids: list[str], model
    ) -> np.ndarray:
        """Mean-pool 1-hop neighbour embeddings (GraphSAGE MEAN aggregator).

        From Hamilton 2017: h_v = σ(W · MEAN(h_v ∪ {h_u, ∀u ∈ N(v)}))
        Simplified: mean of self + available neighbours from ChromaDB.
        """
        collection = self._get_collection()
        existing_ids = set(collection.get(include=[])["ids"])
        aggregated = []

        for i, (vec, eid) in enumerate(zip(vecs, ids)):
            # find 1-hop neighbours in the graph
            rows = self._store._conn.execute(
                """SELECT DISTINCT object FROM triples WHERE subject=? AND predicate NOT IN ('HAS_FILE_TYPE','CONTAINS_KEY','HAS_TAG')
                   UNION
                   SELECT DISTINCT subject FROM triples WHERE object=? AND predicate NOT IN ('HAS_FILE_TYPE','CONTAINS_KEY','HAS_TAG')
                   LIMIT 16""",
                (eid, eid),
            ).fetchall()

            neighbour_ids = [r[0] for r in rows if r[0] in existing_ids and r[0] != eid]

            if not neighbour_ids:
                aggregated.append(vec)
                continue

            # fetch neighbour embeddings from ChromaDB
            result = collection.get(ids=neighbour_ids, include=["embeddings"])
            if result["embeddings"] is None or len(result["embeddings"]) == 0:
                aggregated.append(vec)
                continue

            neighbour_vecs = np.array(result["embeddings"], dtype=np.float32)
            # MEAN aggregation: (self + mean(neighbours)) / 2, then renormalise
            agg = (vec + neighbour_vecs.mean(axis=0)) / 2.0
            norm = np.linalg.norm(agg)
            aggregated.append(agg / norm if norm > 0 else vec)

        return np.array(aggregated, dtype=np.float32)

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

    def _turbo_search(self, entity_id: str, k: int, kind_filter: str | None) -> list[dict]:
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

        out = []
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

    def _chroma_search(self, entity_id: str, k: int, kind_filter: str | None) -> list[dict]:
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
        out = []
        for eid, meta, dist in zip(
            qr["ids"][0], qr["metadatas"][0], qr["distances"][0]
        ):
            if eid == entity_id:
                continue
            out.append({"entity_id": eid, "kind": meta.get("kind"), "score": 1 - dist})
            if len(out) >= k:
                break
        return out

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _get_collection(self):
        if self._collection is None:
            import chromadb
            self._chroma = chromadb.PersistentClient(path=self._chroma_path)
            self._collection = self._chroma.get_or_create_collection(
                name="entities",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection
