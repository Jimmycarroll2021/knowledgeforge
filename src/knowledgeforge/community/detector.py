"""Community detection + LLM summarisation — Phase 7 gap from Edge et al. 2024.

Edge et al. 2024 (arXiv:2404.16130) — GraphRAG global mode:
  1. Build graph from triples
  2. Run Leiden/Louvain community detection
  3. Generate LLM community summaries (cached)
  4. Global queries search community summaries, not just k-hop subgraphs

Uses networkx Louvain (3.x built-in). Summaries cached in SQLite.
Minimum community size filters noise (singleton entities from structural triples).
"""
from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING, Any

import networkx as nx
from networkx.algorithms.community import louvain_communities

if TYPE_CHECKING:
    from ..store.sqlite import SQLiteGraphStore

# Predicates that carry semantic meaning — skip structural noise for community graph
_SEMANTIC_PREDICATES = {
    "PROPOSED_BY", "TYPE_OF", "IMPROVES_ON", "EXTENDS", "EVALUATES_ON",
    "ACHIEVES", "SCALES_TO", "USED_IN", "AUTHORED_BY", "PUBLISHED_IN",
    "RELATED_TO", "DEFINED_AS", "PART_OF", "REQUIRES", "OUTPERFORMS",
    "IMPLEMENTED_IN", "SIMILAR_TO", "SAME_AS", "LINKS_TO",
}

_COMMUNITY_SYSTEM = """You are a knowledge graph analyst. Given a cluster of related entities
and their relationships, write a concise paragraph (3-6 sentences) summarising:
- What domain or topic this cluster represents
- The key entities and how they relate
- Why these entities are grouped together

Be specific. Use the entity names and relationship types from the facts provided.
Output only the summary paragraph — no headers, no bullet points."""


def _call_claude_cli(prompt: str, system: str, model: str) -> str:
    full = f"<system>\n{system}\n</system>\n\n{prompt}"
    r = subprocess.run(
        ["claude", "-p", "--model", model, "-"],
        input=full, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=120,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr[:200])
    return r.stdout.strip()


class CommunityDetector:
    """Detect communities in the knowledge graph and generate LLM summaries.

    From Edge et al. 2024: community summaries enable global queries that
    k-hop local retrieval cannot answer (e.g. "what are the main themes?").

    Usage:
        detector = CommunityDetector(store)
        stats = detector.detect_and_summarise()
        # → communities + summaries written to SQLite
    """

    def __init__(
        self,
        store: "SQLiteGraphStore",
        model: str = "claude-haiku-4-5-20251001",
        min_community_size: int = 3,
        api_key: str | None = None,
    ) -> None:
        self._store = store
        self._model = model
        self._min_size = min_community_size
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._use_cli = not bool(self._api_key)
        self._ensure_tables()

    def detect_and_summarise(self) -> dict[str, Any]:
        """Run Louvain community detection and generate LLM summaries.

        Returns stats: {communities_found, communities_summarised, entities_covered}
        """
        G = self._build_semantic_graph()
        if len(G.nodes) < 2:
            return {"communities_found": 0, "communities_summarised": 0, "entities_covered": 0}

        # Louvain — networkx 3.x built-in (approximates Leiden quality)
        partitions = louvain_communities(G, seed=42, resolution=1.0)

        # Filter to min size, write assignments
        communities = [p for p in partitions if len(p) >= self._min_size]
        self._write_community_assignments(communities)

        # Generate LLM summary for each community
        summarised = 0
        for i, members in enumerate(communities):
            cid = f"community_{i:04d}"
            try:
                summary = self._summarise_community(cid, list(members))
                self._store._conn.execute(
                    "UPDATE communities SET summary=?, summary_model=? WHERE community_id=?",
                    (summary, self._model, cid),
                )
                self._store._conn.commit()
                summarised += 1
            except Exception:
                pass

        entities_covered = sum(len(p) for p in communities)
        return {
            "communities_found": len(communities),
            "communities_summarised": summarised,
            "entities_covered": entities_covered,
            "total_graph_nodes": len(G.nodes),
        }

    def load_summaries(self) -> list[dict[str, Any]]:
        """Load all community summaries from SQLite."""
        rows = self._store._conn.execute(
            """SELECT community_id, member_count, summary
               FROM communities
               WHERE summary IS NOT NULL AND summary != ''
               ORDER BY member_count DESC"""
        ).fetchall()
        return [
            {"community_id": r[0], "member_count": r[1], "summary": r[2]}
            for r in rows
        ]

    def community_stats(self) -> dict[str, Any]:
        row = self._store._conn.execute(
            "SELECT COUNT(*), SUM(member_count) FROM communities"
        ).fetchone()
        return {
            "total_communities": row[0] or 0,
            "total_members": row[1] or 0,
        }

    # ── private ───────────────────────────────────────────────────────────────

    def _build_semantic_graph(self) -> nx.Graph:
        """Build undirected graph from semantic triples only (skip structural noise)."""
        G: nx.Graph = nx.Graph()

        rows = self._store._conn.execute(
            """SELECT subject, predicate, object FROM triples
               WHERE predicate IN ({})""".format(
                ",".join(f"'{p}'" for p in _SEMANTIC_PREDICATES)
            )
        ).fetchall()

        for subj, pred, obj in rows:
            if subj and obj and subj != obj:
                # Weight edges by predicate type — SIMILAR_TO/SAME_AS are strongest
                weight = 2.0 if pred in ("SIMILAR_TO", "SAME_AS", "RELATED_TO") else 1.0
                if G.has_edge(subj, obj):
                    G[subj][obj]["weight"] += weight
                else:
                    G.add_edge(subj, obj, weight=weight)

        return G

    def _write_community_assignments(self, communities: list[set[str]]) -> None:
        """Write community assignments to SQLite, clearing previous."""
        with self._store._conn:
            self._store._conn.execute("DELETE FROM communities")
            self._store._conn.execute("DELETE FROM community_members")

            from ..contracts import now_iso
            ts = now_iso()

            for i, members in enumerate(communities):
                cid = f"community_{i:04d}"
                self._store._conn.execute(
                    "INSERT INTO communities(community_id, member_count, created_at) VALUES (?,?,?)",
                    (cid, len(members), ts),
                )
                self._store._conn.executemany(
                    "INSERT INTO community_members(community_id, entity_id) VALUES (?,?)",
                    [(cid, eid) for eid in members],
                )

    def _summarise_community(self, community_id: str, entity_ids: list[str]) -> str:
        """Generate an LLM summary of a community."""
        # Gather triples for community members
        placeholders = ",".join("?" * len(entity_ids))
        rows = self._store._conn.execute(
            f"""SELECT DISTINCT subject, predicate, object FROM triples
                WHERE (subject IN ({placeholders}) OR object IN ({placeholders}))
                AND predicate NOT IN ('HAS_FILE_TYPE','CONTAINS_KEY','HAS_TAG')
                ORDER BY predicate
                LIMIT 80""",
            entity_ids + entity_ids,
        ).fetchall()

        if not rows:
            return ""

        facts = "\n".join(f"({s})-[{p}]->({o})" for s, p, o in rows)
        prompt = (
            f"Community ID: {community_id}\n"
            f"Entities ({len(entity_ids)}): {', '.join(sorted(entity_ids)[:20])}\n\n"
            f"Relationships:\n{facts}"
        )

        if self._use_cli:
            return _call_claude_cli(prompt, _COMMUNITY_SYSTEM, self._model)

        import anthropic
        client = anthropic.Anthropic(api_key=self._api_key)
        r = client.messages.create(
            model=self._model,
            max_tokens=512,
            system=_COMMUNITY_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        block = r.content[0]
        return block.text.strip() if block.type == "text" else ""

    def _ensure_tables(self) -> None:
        self._store._conn.executescript("""
            CREATE TABLE IF NOT EXISTS communities (
                community_id   TEXT PRIMARY KEY,
                member_count   INTEGER NOT NULL DEFAULT 0,
                summary        TEXT,
                summary_model  TEXT,
                created_at     TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS community_members (
                community_id   TEXT NOT NULL,
                entity_id      TEXT NOT NULL,
                PRIMARY KEY (community_id, entity_id)
            );
            CREATE INDEX IF NOT EXISTS idx_cm_entity ON community_members(entity_id);
        """)
        self._store._conn.commit()
