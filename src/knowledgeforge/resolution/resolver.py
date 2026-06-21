"""Entity resolver — 3-phase pipeline from KF - Entity Resolution.md.

Phase 1: Strong ID match (exact name after normalisation, within same type)
Phase 2: Weak ID match (Jaro-Winkler >= threshold, block by entity type)
Phase 3: Structural match (WCC on SIMILAR edges — catches cross-source dups)

From concepts/Entity Resolution.md:
  - Precision >0.95, Recall >0.90, F1 >0.92
  - False positive merge worse than missed match → conservative threshold 0.85
  - SAME_AS edges, not DELETE — original IDs remain for provenance

Writes to entity_aliases table. Does NOT delete original entities.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import jellyfish

from ..store.sqlite import SQLiteGraphStore

_NORMALISE_RE = re.compile(r"[^a-z0-9]+")

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


class EntityResolver:
    """Runs 3-phase entity resolution on the SQLite graph store.

    Writes SAME_AS edges to entity_aliases table.
    Call .resolve() then read stats or query aliases.
    """

    def __init__(self, store: SQLiteGraphStore, threshold: float = 0.85) -> None:
        self._conn = store._conn
        self._threshold = threshold
        self._conn.executescript(_ALIAS_DDL)
        self._conn.commit()

    def resolve(self) -> dict:
        """Run all phases. Returns merge stats."""
        from ..contracts import now_iso
        ts = now_iso()

        phase1 = self._phase1_exact(ts)
        phase2 = self._phase2_jaro(ts)
        phase3 = self._phase3_structural(ts)

        return {
            "phase1_merges": len(phase1),
            "phase2_merges": len(phase2),
            "phase3_merges": len(phase3),
            "total_merges": len(phase1) + len(phase2) + len(phase3),
        }

    def canonical(self, entity_id: str) -> str:
        """Return canonical ID for an entity (itself if not an alias)."""
        row = self._conn.execute(
            "SELECT canonical_id FROM entity_aliases WHERE alias_id = ?",
            (entity_id,),
        ).fetchone()
        return row[0] if row else entity_id

    def aliases_for(self, canonical_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT alias_id FROM entity_aliases WHERE canonical_id = ?",
            (canonical_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def stats(self) -> dict:
        total = self._conn.execute(
            "SELECT COUNT(*) FROM entity_aliases"
        ).fetchone()[0]
        by_phase = {
            r[0]: r[1]
            for r in self._conn.execute(
                "SELECT phase, COUNT(*) FROM entity_aliases GROUP BY phase"
            ).fetchall()
        }
        return {"total_aliases": total, "by_phase": by_phase}

    # ── phases ────────────────────────────────────────────────────────────────

    def _phase1_exact(self, ts: str) -> list[MergeGroup]:
        """Phase 1: exact normalised name match within same entity type."""
        entities = self._conn.execute(
            "SELECT id, kind FROM entities"
        ).fetchall()

        by_key: dict[tuple[str, str], list[str]] = {}
        for eid, kind in entities:
            norm = _normalise(eid)
            if not norm:
                continue
            key = (kind, norm)
            by_key.setdefault(key, []).append(eid)

        groups: list[MergeGroup] = []
        for (kind, norm), ids in by_key.items():
            if len(ids) < 2:
                continue
            canonical = sorted(ids)[0]
            for alias in ids[1:]:
                self._write_alias(canonical, alias, 1.0, 1, "exact_normalised", ts)
            groups.append(MergeGroup(canonical, ids[1:], 1.0, 1, "exact_normalised"))
        return groups

    def _phase2_jaro(self, ts: str) -> list[MergeGroup]:
        """Phase 2: Jaro-Winkler >= threshold, blocking by entity type."""
        entities = self._conn.execute(
            "SELECT id, kind FROM entities"
        ).fetchall()

        by_kind: dict[str, list[str]] = {}
        for eid, kind in entities:
            by_kind.setdefault(kind, []).append(eid)

        groups: list[MergeGroup] = []
        seen_pairs: set[tuple[str, str]] = set()

        for kind, ids in by_kind.items():
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a, b = ids[i], ids[j]
                    pair = (min(a, b), max(a, b))
                    if pair in seen_pairs:
                        continue
                    score = jellyfish.jaro_winkler_similarity(
                        _normalise(a), _normalise(b)
                    )
                    if score >= self._threshold:
                        seen_pairs.add(pair)
                        canonical = a if len(a) <= len(b) else b
                        alias = b if canonical == a else a
                        existing = self._conn.execute(
                            "SELECT 1 FROM entity_aliases WHERE canonical_id=? AND alias_id=?",
                            (canonical, alias),
                        ).fetchone()
                        if not existing:
                            self._write_alias(canonical, alias, score, 2, "jaro_winkler", ts)
                            groups.append(
                                MergeGroup(canonical, [alias], score, 2, "jaro_winkler")
                            )
        return groups

    def _phase3_structural(self, ts: str) -> list[MergeGroup]:
        """Phase 3: entities connected by SIMILAR_TO → WCC merge candidates."""
        rows = self._conn.execute(
            "SELECT subject, object FROM triples WHERE predicate='SIMILAR_TO'"
        ).fetchall()

        if not rows:
            return []

        # Union-Find for WCC
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent.get(x, x), parent.get(x, x))
                x = parent.get(x, x)
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for subj, obj in rows:
            union(subj, obj)

        components: dict[str, list[str]] = {}
        all_nodes = set()
        for subj, obj in rows:
            all_nodes.update([subj, obj])
        for node in all_nodes:
            root = find(node)
            components.setdefault(root, []).append(node)

        groups: list[MergeGroup] = []
        for root, members in components.items():
            if len(members) < 2:
                continue
            canonical = sorted(members)[0]
            for alias in members:
                if alias == canonical:
                    continue
                existing = self._conn.execute(
                    "SELECT 1 FROM entity_aliases WHERE canonical_id=? AND alias_id=?",
                    (canonical, alias),
                ).fetchone()
                if not existing:
                    self._write_alias(canonical, alias, 0.80, 3, "structural_wcc", ts)
                    groups.append(
                        MergeGroup(canonical, [alias], 0.80, 3, "structural_wcc")
                    )
        return groups

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
