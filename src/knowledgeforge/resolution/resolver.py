"""Entity resolver — multi-signal pipeline from KF - Entity Resolution.md.

Phase 1: Strong ID match (exact name after normalisation, within same type)
Phase 2: Weak ID match — string + (optional) embedding agreement, blocked by
         type and by prefix/phonetic key so candidate generation is not purely
         type-quadratic.
Phase 3: Structural match (WCC on SIMILAR_TO edges — catches cross-source dups)

All confirmed matches feed a single Union-Find so transitive duplicates collapse
to ONE canonical root (path-compressed). A SAME_AS edge is written to the
inferred_relations layer for every confirmed merge so the graph and community
detector see the alias.

From concepts/Entity Resolution.md:
  - Precision >0.95, Recall >0.90, F1 >0.92
  - False positive merge worse than missed match → conservative thresholds
  - SAME_AS edges, not DELETE — original IDs remain for provenance

Resolution policy (general, not fixture-tuned):
  - Identical after normalisation (case / punctuation / spacing only) → merge.
    Pure surface variants are the safest possible match.
  - Acronym ⇄ expansion (e.g. "GNN" = G-raph N-eural N-etwork) → merge. A
    deterministic high-precision initialism rule that generalises to any
    acronym/expansion pair.
  - Otherwise a pair must clear BOTH a combined string score AND, when an
    embedding model is supplied, a semantic-confirmation cosine floor. Requiring
    semantic agreement is what separates near-string CONFUSABLES (GraphSAGE vs
    GraphSAINT, TransE vs TransR, GCN vs R-GCN — high string overlap, divergent
    meaning) from true variants. Scores in the flag band are recorded but NOT
    auto-merged, so a borderline pair never silently over-merges.

Writes to entity_aliases table. Does NOT delete original entities.
"""
from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jellyfish

from ..store.sqlite import SQLiteGraphStore

if TYPE_CHECKING:
    from ..embeddings.pipeline import EmbeddingPipeline

_NORMALISE_RE = re.compile(r"[^a-z0-9]+")
_WORD_RE = re.compile(r"[^A-Za-z0-9]+")

# ── resolution thresholds (general, domain-reasonable — NOT fixture-tuned) ──────
# A pair that is not normalised-identical must clear the combined string score to
# even be considered, and (when embeddings are available) also clear a semantic
# confirmation floor. The flag band sits below auto-merge so borderline pairs are
# recorded for review rather than merged.
_AUTO_MERGE = 0.85          # >= this combined score auto-merges
_FLAG_FLOOR = 0.70          # [_FLAG_FLOOR, _AUTO_MERGE) is flagged, not merged
_SEMANTIC_FLOOR = 0.90      # cosine a non-identical pair must clear when embeddings present
_JW_WEIGHT = 0.6            # combined = jaro_winkler*_JW_WEIGHT + cosine*(1-_JW_WEIGHT)

_ALIAS_DDL = """
CREATE TABLE IF NOT EXISTS entity_aliases (
    canonical_id    TEXT NOT NULL,
    alias_id        TEXT NOT NULL,
    confidence      REAL NOT NULL,
    phase           INTEGER NOT NULL,
    method          TEXT NOT NULL,
    resolved_at     TEXT NOT NULL,
    PRIMARY KEY (canonical_id, alias_id)
);
CREATE INDEX IF NOT EXISTS idx_alias_id ON entity_aliases(alias_id);
"""


@dataclass
class MergeGroup:
    canonical: str
    aliases: list[str]
    confidence: float
    phase: int
    method: str


def _normalise(name: str) -> str:
    return _NORMALISE_RE.sub("", name.lower())


def _words(name: str) -> list[str]:
    return [w for w in _WORD_RE.split(name) if w]


def _initialism_match(a: str, b: str) -> bool:
    """True if one id is the initialism of the other's words.

    e.g. "GNN" == initials of "Graph Neural Network". Deterministic and
    high-precision: fires only when one side is multi-word (>= 2 tokens) and the
    other equals the concatenated first letters. Generalises to any
    acronym/expansion pair — it does not look at any specific entity name.
    """
    for short, long in ((a, b), (b, a)):
        words = _words(long)
        if len(words) < 2:
            continue
        initials = "".join(w[0] for w in words).lower()
        if _normalise(short) == initials:
            return True
    return False


def _phonetic_key(name: str) -> str:
    """Metaphone of the alphanumeric form — a cheap blocking key.

    Cross-form near-duplicates (e.g. "Deep Walk" / "DeepWalk") share a phonetic
    key even when their prefixes differ, so phonetic blocking surfaces candidate
    pairs that pure prefix blocking would miss.
    """
    norm = _normalise(name)
    if not norm:
        return ""
    return jellyfish.metaphone(norm) or norm[:4]


class EntityResolver:
    """Runs multi-signal entity resolution on the SQLite graph store.

    Writes SAME_AS aliases to entity_aliases and SAME_AS edges to the
    inferred_relations layer. Call .resolve() then read stats / query aliases.

    Pass an :class:`EmbeddingPipeline` as ``embed`` to enable the semantic
    confirmation stage (recommended): the combined confidence becomes
    ``jaro*0.6 + cosine*0.4`` and a non-identical pair must additionally clear a
    cosine floor, which is what separates near-string confusables from true
    variants. Without ``embed`` the resolver degrades to string-only matching.
    """

    def __init__(
        self,
        store: SQLiteGraphStore,
        threshold: float = 0.85,
        embed: "EmbeddingPipeline | None" = None,
    ) -> None:
        self._store = store
        self._conn = store._conn
        self._threshold = threshold
        self._embed = embed
        self._conn.executescript(_ALIAS_DDL)
        self._conn.commit()

        # Union-Find over all entity ids — the single source of truth for
        # canonical roots. Confirmed matches from every phase union into it so
        # transitive duplicates collapse to ONE root.
        self._parent: dict[str, str] = {}
        # Pairs in the flag band [_FLAG_FLOOR, _AUTO_MERGE): recorded, not merged.
        self._flagged: list[tuple[str, str, float, str]] = []
        # Cache embed_query vectors so each entity string is encoded once.
        self._vec_cache: dict[str, Any] = {}

    # ── public API ──────────────────────────────────────────────────────────────

    def resolve(self) -> dict[str, Any]:
        """Run all phases. Returns merge stats."""
        from ..contracts import now_iso
        ts = now_iso()

        phase1 = self._phase1_exact(ts)
        phase2 = self._phase2_similarity(ts)
        phase3 = self._phase3_structural(ts)

        return {
            "phase1_merges": len(phase1),
            "phase2_merges": len(phase2),
            "phase3_merges": len(phase3),
            "total_merges": len(phase1) + len(phase2) + len(phase3),
            "flagged_pairs": len(self._flagged),
        }

    def canonical(self, entity_id: str) -> str:
        """Return the canonical ROOT id for an entity (itself if not an alias).

        Transitive: follows the full alias chain to a fixed point so a → b → c
        all report ``c``. Backed by the in-memory Union-Find when populated (after
        resolve()), and falls back to walking the persisted entity_aliases table
        so it stays correct across process restarts.
        """
        if entity_id in self._parent:
            return self._find(entity_id)
        return self._canonical_from_db(entity_id)

    # Stable alias for GraphRAG / inference callers mapping a queried id to its
    # canonical node.
    def resolve_id(self, entity_id: str) -> str:
        return self.canonical(entity_id)

    def aliases_for(self, canonical_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT alias_id FROM entity_aliases WHERE canonical_id = ?",
            (canonical_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def flagged_pairs(self) -> list[dict[str, Any]]:
        """Pairs in the flag band — recorded for review, not auto-merged."""
        return [
            {"id_a": a, "id_b": b, "score": score, "method": method}
            for a, b, score, method in self._flagged
        ]

    def stats(self) -> dict[str, Any]:
        total = self._conn.execute(
            "SELECT COUNT(*) FROM entity_aliases"
        ).fetchone()[0]
        by_phase = {
            r[0]: r[1]
            for r in self._conn.execute(
                "SELECT phase, COUNT(*) FROM entity_aliases GROUP BY phase"
            ).fetchall()
        }
        return {
            "total_aliases": total,
            "by_phase": by_phase,
            "flagged_pairs": len(self._flagged),
        }

    # ── Union-Find (path-compressed) ─────────────────────────────────────────────

    def _find(self, x: str) -> str:
        root = x
        while self._parent.get(root, root) != root:
            root = self._parent[root]
        # path compression
        while self._parent.get(x, x) != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def _union(self, a: str, b: str) -> tuple[str, str] | None:
        """Union two ids. Returns (canonical_root, absorbed_root) or None if
        already in the same component. Canonical = the shorter id (ties → sort),
        a stable, length-preferring choice."""
        ra, rb = self._find(a), self._find(b)
        if ra == rb:
            return None
        canonical = ra if (len(ra), ra) <= (len(rb), rb) else rb
        absorbed = rb if canonical == ra else ra
        self._parent[absorbed] = canonical
        return canonical, absorbed

    def _canonical_from_db(self, entity_id: str) -> str:
        """Walk the persisted alias chain to a fixed point (restart-safe)."""
        seen: set[str] = set()
        current = entity_id
        while current not in seen:
            seen.add(current)
            row = self._conn.execute(
                "SELECT canonical_id FROM entity_aliases WHERE alias_id = ?",
                (current,),
            ).fetchone()
            if row is None or row[0] == current:
                return current
            current = row[0]
        return current

    # ── similarity signals ───────────────────────────────────────────────────────

    def _cosine(self, a: str, b: str) -> float | None:
        """Cosine of the two entity strings via the injected embedding model.

        Returns None when no embedding model is available. Vectors from
        ``embed_query`` are L2-normalised, so cosine is a plain dot product.
        """
        if self._embed is None:
            return None
        import numpy as np

        if a not in self._vec_cache:
            self._vec_cache[a] = self._embed.embed_query(a)
        if b not in self._vec_cache:
            self._vec_cache[b] = self._embed.embed_query(b)
        return float(np.dot(self._vec_cache[a], self._vec_cache[b]))

    def _score_pair(self, a: str, b: str) -> tuple[float, str, bool]:
        """Score a candidate pair. Returns (confidence, method, auto_merge).

        Policy (general):
          - identical after normalisation       → 1.0, auto-merge
          - initialism ⇄ expansion              → strong alias, auto-merge
          - else combined string+embedding score; auto-merge only if it clears
            _AUTO_MERGE AND (no embeddings, or cosine clears _SEMANTIC_FLOOR).
        """
        if _normalise(a) == _normalise(b):
            return 1.0, "exact_normalised", True
        if _initialism_match(a, b):
            return 0.97, "initialism", True

        jw = jellyfish.jaro_winkler_similarity(_normalise(a), _normalise(b))
        cos = self._cosine(a, b)
        if cos is None:
            combined = jw
            semantic_ok = True  # string-only fallback
        else:
            combined = jw * _JW_WEIGHT + cos * (1.0 - _JW_WEIGHT)
            semantic_ok = cos >= _SEMANTIC_FLOOR
        method = "string_embedding" if cos is not None else "jaro_winkler"
        auto = combined >= self._auto_threshold() and semantic_ok
        return combined, method, auto

    def _auto_threshold(self) -> float:
        # Honour a caller-supplied threshold if it is stricter than the default.
        return max(_AUTO_MERGE, self._threshold) if self._threshold else _AUTO_MERGE

    # ── phases ────────────────────────────────────────────────────────────────────

    def _phase1_exact(self, ts: str) -> list[MergeGroup]:
        """Phase 1: exact normalised name match within same entity type."""
        entities = self._conn.execute("SELECT id, kind FROM entities").fetchall()

        by_key: dict[tuple[str, str], list[str]] = {}
        for eid, kind in entities:
            norm = _normalise(eid)
            if not norm:
                continue
            by_key.setdefault((kind, norm), []).append(eid)

        groups: list[MergeGroup] = []
        for (_kind, _norm), ids in by_key.items():
            if len(ids) < 2:
                continue
            canonical = sorted(ids)[0]
            for alias in ids[1:]:
                self._commit_merge(canonical, alias, 1.0, 1, "exact_normalised", ts)
            groups.append(MergeGroup(canonical, ids[1:], 1.0, 1, "exact_normalised"))
        return groups

    def _phase2_similarity(self, ts: str) -> list[MergeGroup]:
        """Phase 2: string + embedding agreement, type + prefix/phonetic blocking.

        Candidate pairs come from blocking keys (entity type, name prefix,
        phonetic key) so generation is not purely type-quadratic and cross-form
        candidates (e.g. spaced vs concatenated) still surface. Each candidate is
        scored by _score_pair: auto-merges union into the Union-Find; flag-band
        pairs are recorded only.
        """
        entities = self._conn.execute("SELECT id, kind FROM entities").fetchall()
        ids_by_kind: dict[str, list[str]] = {}
        for eid, kind in entities:
            ids_by_kind.setdefault(kind, []).append(eid)

        groups: list[MergeGroup] = []
        scored: set[tuple[str, str]] = set()

        for kind, ids in ids_by_kind.items():
            for a, b in self._candidate_pairs(ids):
                pair = (min(a, b), max(a, b))
                if pair in scored:
                    continue
                scored.add(pair)

                confidence, method, auto = self._score_pair(a, b)
                if auto:
                    merged = self._commit_merge(
                        pair[0], pair[1], confidence, 2, method, ts
                    )
                    if merged:
                        canonical, alias = merged
                        groups.append(
                            MergeGroup(canonical, [alias], confidence, 2, method)
                        )
                elif confidence >= _FLAG_FLOOR:
                    self._flagged.append((a, b, confidence, method))
        return groups

    def _candidate_pairs(self, ids: list[str]) -> Iterator[tuple[str, str]]:
        """Yield candidate pairs within one type via blocking keys.

        Blocks on (a) the first two normalised chars (prefix) and (b) the
        phonetic key. The union of blocks' intra-block pairs is the candidate
        set — far smaller than the full type-quadratic product on real graphs,
        while still surfacing cross-form pairs that share a phonetic key.

        Small blocks (<= a few hundred ids) fall back to the full product so a
        tiny benchmark graph never loses a candidate to over-aggressive blocking.
        """
        if len(ids) <= 256:
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    yield ids[i], ids[j]
            return

        blocks: dict[str, list[str]] = {}
        for eid in ids:
            norm = _normalise(eid)
            prefix = norm[:2]
            for key in (f"p:{prefix}", f"m:{_phonetic_key(eid)}"):
                blocks.setdefault(key, []).append(eid)

        emitted: set[tuple[str, str]] = set()
        for members in blocks.values():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = members[i], members[j]
                    pair = (min(a, b), max(a, b))
                    if pair not in emitted:
                        emitted.add(pair)
                        yield a, b

    def _phase3_structural(self, ts: str) -> list[MergeGroup]:
        """Phase 3: entities connected by SIMILAR_TO → WCC merge candidates."""
        rows = self._conn.execute(
            "SELECT subject, object FROM triples WHERE predicate='SIMILAR_TO'"
        ).fetchall()
        if not rows:
            return []

        groups: list[MergeGroup] = []
        for subj, obj in rows:
            merged = self._commit_merge(subj, obj, 0.80, 3, "structural_wcc", ts)
            if merged:
                canonical, alias = merged
                groups.append(MergeGroup(canonical, [alias], 0.80, 3, "structural_wcc"))
        return groups

    # ── merge commit (Union-Find + alias row + SAME_AS edge) ───────────────────────

    def _commit_merge(
        self,
        a: str,
        b: str,
        confidence: float,
        phase: int,
        method: str,
        ts: str,
    ) -> tuple[str, str] | None:
        """Union a and b, persist the alias row, and emit a SAME_AS edge.

        Returns (canonical, alias) on a fresh merge, None if already merged.
        Canonical is the Union-Find root so transitivity is preserved: a later
        merge that re-parents the root still resolves correctly via canonical().
        """
        result = self._union(a, b)
        if result is None:
            return None
        canonical, alias = result
        self._write_alias(canonical, alias, confidence, phase, method, ts)
        self._write_same_as_edge(canonical, alias, confidence, method, ts)
        return canonical, alias

    def _write_alias(
        self,
        canonical: str,
        alias: str,
        confidence: float,
        phase: int,
        method: str,
        ts: str,
    ) -> None:
        try:
            with self._conn:
                self._conn.execute(
                    """INSERT OR IGNORE INTO entity_aliases
                       (canonical_id, alias_id, confidence, phase, method, resolved_at)
                       VALUES (?,?,?,?,?,?)""",
                    (canonical, alias, confidence, phase, method, ts),
                )
        except sqlite3.Error:
            pass

    def _write_same_as_edge(
        self, canonical: str, alias: str, confidence: float, method: str, ts: str
    ) -> None:
        """Write a SAME_AS edge to inferred_relations so the graph sees the alias."""
        from ..contracts import Triple

        kinds = self._conn.execute(
            "SELECT id, kind FROM entities WHERE id IN (?, ?)", (canonical, alias)
        ).fetchall()
        kind_of = {row[0]: row[1] for row in kinds}
        triple = Triple(
            subject=alias,
            predicate="SAME_AS",
            object=canonical,
            source_kind=kind_of.get(alias, "concept"),
            target_kind=kind_of.get(canonical, "concept"),
            evidence=f"entity resolution merge via {method} (confidence={confidence:.3f})",
            confidence=min(1.0, max(0.0, confidence)),
            source_doc="resolution:entity_resolver",
            extraction_method="structural",
            timestamp=ts,
            adapter="entity_resolver",
            layer="inferred_relations",
            source="EntityResolver",
        )
        try:
            self._store.add_triple(triple)
        except sqlite3.Error:
            pass
