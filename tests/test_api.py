"""API integration tests for KnowledgeForge FastAPI app.

Uses a fresh SQLite DB per test via tmp_path (never touches data/graph.db).
GraphRAG._call_llm is patched to avoid real LLM calls in the query tests.
"""
import os
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


def _make_client(tmp_path: Path) -> TestClient:
    """Create a TestClient with isolated tmp DB + embeddings path."""
    os.environ["KF_DB_PATH"] = str(tmp_path / "test.db")
    os.environ["KF_EMBEDDINGS_PATH"] = str(tmp_path / "embeddings")
    # Import after env vars are set so lifespan picks them up
    from knowledgeforge.api.main import app
    return TestClient(app)


# ── /health ──────────────────────────────────────────────────────────────────

def test_health(tmp_path):
    with _make_client(tmp_path) as client:
        r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "entities" in data
    assert "triples" in data
    assert "embedded_entities" in data


# ── /graph/stats ─────────────────────────────────────────────────────────────

def test_graph_stats(tmp_path):
    with _make_client(tmp_path) as client:
        r = client.get("/graph/stats")
    assert r.status_code == 200
    data = r.json()
    assert "entities" in data
    assert "triples" in data
    assert "by_layer" in data
    assert "top_predicates" in data


# ── /ingest ───────────────────────────────────────────────────────────────────

def test_ingest_with_markdown(tmp_path):
    src = tmp_path / "docs"
    src.mkdir()
    (src / "graphsage.md").write_text(
        "# GraphSAGE\nGraphSAGE is an inductive representation learning algorithm.\n",
        encoding="utf-8",
    )
    with _make_client(tmp_path) as client:
        r = client.post("/ingest", json={"source": str(src), "adapter": "universal"})
    assert r.status_code == 200
    data = r.json()
    assert data["triples_added"] >= 0
    assert "documents_scanned" in data
    assert "errors" in data


def test_ingest_bad_path(tmp_path):
    with _make_client(tmp_path) as client:
        r = client.post(
            "/ingest",
            json={"source": str(tmp_path / "nonexistent_dir"), "adapter": "universal"},
        )
    assert r.status_code == 400


# ── /query ────────────────────────────────────────────────────────────────────

def test_query(tmp_path):
    fake_answer = "The graph does not contain evidence for this."
    with patch(
        "knowledgeforge.inference.graphrag.GraphRAG._call_llm",
        return_value=fake_answer,
    ):
        with _make_client(tmp_path) as client:
            r = client.post("/query", json={"question": "what is a test?"})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert "cited_triples" in data
    assert isinstance(data["cited_triples"], list)
    assert "anchor_entities" in data
    assert isinstance(data["anchor_entities"], list)
    assert "subgraph_size" in data


# ── /graph/node/<id> ─────────────────────────────────────────────────────────

def test_graph_node_not_found(tmp_path):
    with _make_client(tmp_path) as client:
        r = client.get("/graph/node/nonexistent")
    assert r.status_code == 404


# ── /graph/provenance/<id> ───────────────────────────────────────────────────

def test_graph_provenance_not_found(tmp_path):
    with _make_client(tmp_path) as client:
        r = client.get("/graph/provenance/nonexistent")
    assert r.status_code == 404


# ── /similar/<id> ────────────────────────────────────────────────────────────

def test_similar_unknown_entity_returns_empty(tmp_path):
    with _make_client(tmp_path) as client:
        r = client.get("/similar/nonexistent")
    assert r.status_code == 200
    data = r.json()
    assert data["results"] == []
