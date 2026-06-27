"""Hierarchical Leiden community detection + DRIFT GraphRAG mode.

Covers the v2 "Hierarchical GraphRAG" spike:
  - detector builds MECE community levels (level 0 coarse -> deeper finer) with
    correct parent linkage, via seeded Leiden (deterministic)
  - DRIFT query mode fuses community themes with local entity retrieval and
    still grounds only on trusted layers (llm_hypotheses excluded)

LLM calls are monkeypatched — deterministic and offline.
"""

from __future__ import annotations

import itertools

import knowledgeforge.community.detector as detector_mod
from knowledgeforge.contracts import Triple, now_iso
from knowledgeforge.inference.graphrag import GraphRAG
from knowledgeforge.store.sqlite import SQLiteGraphStore


def _add(
    store: SQLiteGraphStore, s: str, p: str, o: str, layer: str = "source_facts", conf: float = 0.9
) -> None:
    store.add_triple(
        Triple(
            subject=s,
            predicate=p,
            object=o,
            source_kind="Concept",
            target_kind="Concept",
            evidence="e",
            confidence=conf,
            source_doc="d",
            extraction_method="test",
            timestamp=now_iso(),
            adapter="test",
            layer=layer,
        )
    )


def _nested_graph(store: SQLiteGraphStore) -> None:
    """Two super-clusters, each two 4-cliques, weakly bridged.

    Coarse resolution -> 2 communities of 8; finer -> 4 of 4. Exercises the
    coarse->fine recursion + parent linkage.
    """
    cliques = {
        "a1": [f"A{i}" for i in range(1, 5)],
        "a2": [f"A{i}" for i in range(5, 9)],
        "b1": [f"B{i}" for i in range(1, 5)],
        "b2": [f"B{i}" for i in range(5, 9)],
    }
    for nodes in cliques.values():
        for x, y in itertools.combinations(nodes, 2):
            _add(store, x, "SIMILAR_TO", y)  # intra-clique (weight 2.0)
    _add(store, "A1", "LINKS_TO", "A5")  # a1<->a2 weak bridge
    _add(store, "B1", "LINKS_TO", "B5")  # b1<->b2 weak bridge
    _add(store, "A1", "LINKS_TO", "B1")  # super A<->B weak bridge


def test_hierarchical_levels(tmp_path, monkeypatch):
    store = SQLiteGraphStore(tmp_path / "hier.db")
    _nested_graph(store)
    monkeypatch.setattr(detector_mod, "_call_claude_cli", lambda p, s, m: "summary.")

    det = detector_mod.CommunityDetector(
        store,
        min_community_size=2,
        base_resolution=0.1,
        resolution_step=4.0,
        split_threshold=4,
        max_levels=2,
    )
    stats = det.detect_and_summarise()

    # Two genuine levels formed: coarse top + finer children.
    assert stats["levels"] == 2
    assert stats["communities_found"] == 6
    assert stats["communities_summarised"] == 6
    assert stats["entities_covered"] == 16

    rows = store._conn.execute(
        "SELECT community_id, level, parent_community_id, member_count FROM communities"
    ).fetchall()
    by_id = {r[0]: {"level": r[1], "parent": r[2], "n": r[3]} for r in rows}

    level0 = [c for c, v in by_id.items() if v["level"] == 0]
    level1 = [c for c, v in by_id.items() if v["level"] == 1]
    assert len(level0) == 2 and all(by_id[c]["n"] == 8 for c in level0)
    assert len(level1) == 4 and all(by_id[c]["n"] == 4 for c in level1)

    # Parent integrity: every child points to an existing coarser (lower-level) parent.
    for cid, v in by_id.items():
        if v["parent"] is None:
            assert v["level"] == 0
        else:
            assert v["parent"] in by_id
            assert by_id[v["parent"]]["level"] < v["level"]

    # Summaries carry level + parent for level-aware consumption.
    summaries = det.load_summaries()
    assert len(summaries) == 6
    assert all("level" in s and "parent_community_id" in s for s in summaries)
    store.close()


def test_flat_when_no_oversized(tmp_path, monkeypatch):
    """A small graph below the split threshold stays a single level."""
    store = SQLiteGraphStore(tmp_path / "flat.db")
    _add(store, "GraphSAGE", "IMPROVES_ON", "GCN")
    _add(store, "GraphSAGE", "TYPE_OF", "Inductive Learning")
    _add(store, "GCN", "RELATED_TO", "Inductive Learning")
    monkeypatch.setattr(detector_mod, "_call_claude_cli", lambda p, s, m: "summary.")

    det = detector_mod.CommunityDetector(store, min_community_size=2)
    stats = det.detect_and_summarise()
    assert stats["communities_found"] >= 1
    assert stats["levels"] == 1
    assert all(s["level"] == 0 for s in det.load_summaries())
    store.close()


def test_empty_graph_stats_shape(tmp_path):
    """A sub-2-node semantic graph still returns the full documented stats shape."""
    store = SQLiteGraphStore(tmp_path / "empty.db")
    # Only a structural (non-semantic) triple → semantic graph has < 2 nodes.
    _add(store, "doc.md", "HAS_FILE_TYPE", "markdown")
    det = detector_mod.CommunityDetector(store, min_community_size=2)
    stats = det.detect_and_summarise()
    for key in (
        "communities_found",
        "communities_summarised",
        "entities_covered",
        "total_graph_nodes",
        "levels",
    ):
        assert key in stats
    assert stats["communities_found"] == 0
    store.close()


def test_drift_mode(tmp_path, monkeypatch):
    """DRIFT fuses community themes + local facts, grounding on trusted layers only."""
    store = SQLiteGraphStore(tmp_path / "drift.db")
    _add(store, "GraphSAGE", "IMPROVES_ON", "GCN")
    _add(store, "GraphSAGE", "TYPE_OF", "Inductive Learning")
    _add(store, "GCN", "RELATED_TO", "Inductive Learning")
    # Untrusted hypothesis that must NOT reach the grounding context.
    _add(
        store,
        "GraphSAGE",
        "DEFINED_AS",
        "A FABRICATED HALLUCINATION",
        layer="llm_hypotheses",
        conf=0.3,
    )

    monkeypatch.setattr(
        detector_mod,
        "_call_claude_cli",
        lambda p, s, m: "A cluster of inductive graph-learning methods (GraphSAGE, GCN).",
    )
    detector_mod.CommunityDetector(store, min_community_size=2).detect_and_summarise()

    captured: dict[str, str] = {}

    def fake_llm(self, question, facts, system=None):
        captured["facts"] = facts
        captured["system"] = system or ""
        return "GraphSAGE improves on GCN."

    monkeypatch.setattr(GraphRAG, "_call_llm", fake_llm)
    rag = GraphRAG(store)
    ans = rag.ask("How does GraphSAGE relate to GCN?", mode="drift")

    assert ans["mode"] == "drift"
    assert ans["communities_used"] >= 1
    assert ans["subgraph_size"] > 0
    assert ans["answer"]
    # Trusted-layer grounding preserved through DRIFT's local expansion.
    assert "FABRICATED HALLUCINATION" not in captured["facts"]
    # The combined context carries both theme context and the DRIFT system prompt.
    assert "DRIFT" in captured["system"]
    assert "Community themes" in captured["facts"]
    store.close()
