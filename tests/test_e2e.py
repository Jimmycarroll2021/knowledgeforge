"""End-to-end pipeline test — the full KnowledgeForge spine on a tiny fixture vault.

Exercises ingest -> resolve -> community detection -> local GraphRAG retrieval
as one wired flow, deterministically and offline (LLM calls are monkeypatched).
Asserts the cross-layer invariants that the unit tests can't: that resolution's
SAME_AS edges land in the inferred_relations layer, that communities form over
the ingested link graph, and that GraphRAG grounding EXCLUDES the untrusted
llm_hypotheses layer.

The learned-GraphSAGE embedding math is covered by tests/test_embeddings.py;
a model-dependent embedding smoke lives in test_e2e_embedding (skipped if the
sentence-transformer can't be loaded, e.g. offline CI).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from knowledgeforge.adapter.vault import VaultAdapter
from knowledgeforge.contracts import Triple, now_iso
from knowledgeforge.inference.graphrag import GraphRAG
from knowledgeforge.pipeline import ForgePipeline
from knowledgeforge.resolution.resolver import EntityResolver
from knowledgeforge.store.sqlite import SQLiteGraphStore


def _make_vault(root: Path) -> None:
    """A small, densely cross-linked graph-ML vault."""
    (root / "algorithms").mkdir(parents=True)
    (root / "concepts").mkdir(parents=True)
    (root / "algorithms" / "GraphSAGE.md").write_text(
        "# GraphSAGE\n\n"
        "GraphSAGE is an [[Inductive Learning]] method. It improves on [[GCN]].\n\n"
        "## Aggregator\nMean, LSTM and pooling aggregators.\n",
        encoding="utf-8",
    )
    (root / "algorithms" / "GCN.md").write_text(
        "# GCN\n\nGraph Convolutional Network. Related to [[GraphSAGE]] and [[Inductive Learning]].\n\n"
        "## Core Equation\nSpectral motivation.\n",
        encoding="utf-8",
    )
    (root / "concepts" / "Inductive Learning.md").write_text(
        "# Inductive Learning\n\nGeneralises to unseen nodes. Used by [[GraphSAGE]].\n",
        encoding="utf-8",
    )


def test_e2e_pipeline(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    _make_vault(vault)
    db = tmp_path / "e2e.db"

    # ── 1. INGEST ───────────────────────────────────────────────────────────
    store = SQLiteGraphStore(db)
    pipeline = ForgePipeline(store)
    result = pipeline.run(VaultAdapter(), vault)
    assert result.triples_added > 0
    assert result.documents_scanned == 3

    stats = store.stats()
    assert stats["entities"] > 0
    assert stats["by_layer"].get("source_facts", 0) == stats["triples"]  # all source facts pre-resolve

    # ── 2. RESOLVE — SAME_AS edges must land in inferred_relations layer ──────
    resolver = EntityResolver(store)
    resolver.resolve()
    # GraphSAGE is referenced as both a doc and a wiki-link target → canonical() resolves.
    assert resolver.canonical("GraphSAGE")  # does not raise

    after = store.stats()
    same_as = store.query(predicate="SAME_AS", limit=1000)
    if same_as:  # if any aliases were found, they must be isolated in inferred_relations
        assert after["by_layer"].get("inferred_relations", 0) >= len(same_as)
        assert all(t["layer"] == "inferred_relations" for t in same_as)

    # ── 3. COMMUNITY DETECTION (Louvain) — mock the LLM summariser ────────────
    import knowledgeforge.community.detector as detector_mod
    monkeypatch.setattr(
        detector_mod, "_call_claude_cli",
        lambda prompt, system, model: "A cluster of related graph-ML methods.",
    )
    detector = detector_mod.CommunityDetector(store, min_community_size=2)
    cstats = detector.detect_and_summarise()
    assert cstats["communities_found"] >= 1
    summaries = detector.load_summaries()
    assert len(summaries) == cstats["communities_summarised"]

    # ── 4. LOCAL GRAPHRAG — mock the answer LLM; verify grounded retrieval ────
    # Plant an untrusted hypothesis that must NOT appear in grounded evidence.
    store.add_triple(Triple(
        subject="GraphSAGE", predicate="DEFINED_AS", object="A FABRICATED HALLUCINATION",
        source_kind="Algorithm", target_kind="Value", evidence="hallucinated",
        confidence=0.3, source_doc="llm", extraction_method="llm",
        timestamp=now_iso(), adapter="llm", layer="llm_hypotheses",
    ))

    captured = {}

    def fake_llm(self, question, facts, system=None):
        captured["facts"] = facts
        return "GraphSAGE is an inductive learning method that improves on GCN."

    monkeypatch.setattr(GraphRAG, "_call_llm", fake_llm)
    rag = GraphRAG(store)
    answer = rag.ask("What is GraphSAGE?", mode="local")

    assert answer["mode"] == "local"
    assert answer["anchor_entities"], "should find at least one anchor"
    assert answer["subgraph_size"] > 0
    assert answer["answer"]
    # The untrusted llm_hypotheses fact must be excluded from the grounding context.
    assert "FABRICATED HALLUCINATION" not in captured.get("facts", "")

    store.close()


def test_e2e_embedding(tmp_path):
    """Real learned-GraphSAGE embedding smoke — skipped if the model can't load."""
    sentence_transformers = pytest.importorskip("sentence_transformers")
    del sentence_transformers

    vault = tmp_path / "vault"
    _make_vault(vault)
    db = tmp_path / "e2e_emb.db"
    store = SQLiteGraphStore(db)
    ForgePipeline(store).run(VaultAdapter(), vault)

    from knowledgeforge.embeddings.pipeline import EmbeddingPipeline
    pipe = EmbeddingPipeline(store, chroma_path=str(tmp_path / "emb"))
    try:
        res = pipe.embed_all()
    except Exception as exc:  # offline / model download failure → skip, don't fail CI
        pytest.skip(f"embedding model unavailable: {exc}")

    assert res["entities_embedded"] > 0
    assert pipe.stats()["embedded_entities"] == res["entities_embedded"]
    store.close()
