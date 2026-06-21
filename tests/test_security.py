"""Security/hardening tests for the KnowledgeForge API.

All hardening is config-gated. These tests set the relevant env var, build a
TestClient, and assert behaviour. The ``_clean_security_env`` fixture guarantees
KF_API_KEY / KF_RATE_LIMIT never leak into the other test modules.

The security middleware reads env per-request, so setting os.environ before a
request is sufficient — no importlib.reload needed.
"""
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

LLM_PATCH = "knowledgeforge.inference.graphrag.GraphRAG._call_llm"


@pytest.fixture(autouse=True)
def _clean_security_env():
    """Ensure auth/rate-limit are unset before and after each test here."""
    for key in ("KF_API_KEY", "KF_RATE_LIMIT"):
        os.environ.pop(key, None)
    yield
    for key in ("KF_API_KEY", "KF_RATE_LIMIT"):
        os.environ.pop(key, None)


def _make_client(tmp_path: Path) -> TestClient:
    os.environ["KF_DB_PATH"] = str(tmp_path / "test.db")
    os.environ["KF_EMBEDDINGS_PATH"] = str(tmp_path / "embeddings")
    from knowledgeforge.api.main import app
    return TestClient(app)


# ── API-key auth (KF_API_KEY set) ─────────────────────────────────────────────

def test_auth_required_query_no_key_401(tmp_path):
    os.environ["KF_API_KEY"] = "secret-key"
    with patch(LLM_PATCH, return_value="answer"):
        with _make_client(tmp_path) as client:
            r = client.post("/query", json={"question": "x"})
    assert r.status_code == 401


def test_auth_correct_key_not_401(tmp_path):
    os.environ["KF_API_KEY"] = "secret-key"
    with patch(LLM_PATCH, return_value="answer"):
        with _make_client(tmp_path) as client:
            r = client.post(
                "/query",
                json={"question": "x"},
                headers={"X-API-Key": "secret-key"},
            )
    assert r.status_code != 401


def test_auth_health_exempt(tmp_path):
    os.environ["KF_API_KEY"] = "secret-key"
    with _make_client(tmp_path) as client:
        r = client.get("/health")  # no key
    assert r.status_code == 200


def test_auth_docs_exempt(tmp_path):
    os.environ["KF_API_KEY"] = "secret-key"
    with _make_client(tmp_path) as client:
        r = client.get("/openapi.json")  # no key
    assert r.status_code == 200


# ── Dev mode (KF_API_KEY unset) ───────────────────────────────────────────────

def test_dev_mode_no_key_open(tmp_path):
    # _clean_security_env guarantees KF_API_KEY is unset.
    with patch(LLM_PATCH, return_value="answer"):
        with _make_client(tmp_path) as client:
            r = client.post("/query", json={"question": "x"})
    assert r.status_code != 401


# ── Rate limiting (KF_RATE_LIMIT set) ─────────────────────────────────────────

def test_rate_limit_429(tmp_path):
    os.environ["KF_RATE_LIMIT"] = "2"
    with patch(LLM_PATCH, return_value="answer"):
        with _make_client(tmp_path) as client:
            r1 = client.post("/query", json={"question": "x"})
            r2 = client.post("/query", json={"question": "x"})
            r3 = client.post("/query", json={"question": "x"})
    assert r1.status_code != 429
    assert r2.status_code != 429
    assert r3.status_code == 429
    assert "Retry-After" in r3.headers


def test_rate_limit_exempts_health(tmp_path):
    os.environ["KF_RATE_LIMIT"] = "1"
    with _make_client(tmp_path) as client:
        for _ in range(5):
            r = client.get("/health")
            assert r.status_code == 200


# ── Request-ID (always on) ────────────────────────────────────────────────────

def test_request_id_header_present(tmp_path):
    with _make_client(tmp_path) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.headers.get("X-Request-ID")
