"""Fast, model-free unit tests for the learned GraphSAGE aggregator.

These exercise the pure math methods (`_build_adjacency`, `_train_aggregator`,
`_apply_aggregator`) with small injected random base vectors — the
sentence-transformer is never loaded, so the tests run in well under a second.
"""
from __future__ import annotations

import numpy as np
import pytest

from knowledgeforge.embeddings.pipeline import EmbeddingPipeline, _EMBED_DIM


# Tiny synthetic graph: 6 nodes, two connected components.
#   a — b — c   (path: a-b, b-c)
#   d — e       (edge d-e)
#   f           (isolated)
_NODES = ["a", "b", "c", "d", "e", "f"]
_ADJACENCY = {
    "a": ["b"],
    "b": ["a", "c"],
    "c": ["b"],
    "d": ["e"],
    "e": ["d"],
    "f": [],
}


def _pipeline(tmp_path) -> EmbeddingPipeline:
    """Build a pipeline whose store is never touched (we inject the math)."""
    p = EmbeddingPipeline.__new__(EmbeddingPipeline)
    p._chroma_path = str(tmp_path)
    p._w_path = tmp_path / "graphsage_w.npy"
    # Mirror the relevant defaults from __init__ without a real store.
    p._store = None
    p._model = None
    p._chroma = None
    p._collection = None
    p._turbo = None
    p._turbo_ids = []
    return p


def _base_vecs(dim: int = _EMBED_DIM) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)  # independent of the trainer's seed
    return {n: rng.normal(0, 1, size=dim).astype(np.float32) for n in _NODES}


def test_training_loss_decreases(tmp_path):
    """The aggregator learns: last-step loss < first-step loss."""
    p = _pipeline(tmp_path)
    base = _base_vecs()
    W = p._train_aggregator(base, _ADJACENCY)

    losses = p._last_train_losses
    assert len(losses) >= 2, "expected per-step losses to be recorded"
    assert losses[-1] < losses[0], (
        f"loss did not decrease: first={losses[0]:.4f} last={losses[-1]:.4f}"
    )
    assert W.shape == (_EMBED_DIM, 2 * _EMBED_DIM)


def test_determinism(tmp_path):
    """Same seed + same inputs → identical W and identical aggregated vectors."""
    base = _base_vecs()

    p1 = _pipeline(tmp_path / "run1")
    (tmp_path / "run1").mkdir()
    p1._w_path = tmp_path / "run1" / "graphsage_w.npy"
    W1 = p1._train_aggregator(base, _ADJACENCY)
    agg1 = p1._apply_aggregator(base, _ADJACENCY, W1)

    p2 = _pipeline(tmp_path / "run2")
    (tmp_path / "run2").mkdir()
    p2._w_path = tmp_path / "run2" / "graphsage_w.npy"
    W2 = p2._train_aggregator(base, _ADJACENCY)
    agg2 = p2._apply_aggregator(base, _ADJACENCY, W2)

    np.testing.assert_array_equal(W1, W2)
    for n in _NODES:
        np.testing.assert_array_equal(agg1[n], agg2[n])


def test_structure_injection_and_isolated_fallback(tmp_path):
    """Connected nodes shift away from their base; isolated node is just L2-norm."""
    p = _pipeline(tmp_path)
    base = _base_vecs()
    W = p._train_aggregator(base, _ADJACENCY)
    agg = p._apply_aggregator(base, _ADJACENCY, W)

    # Node 'b' has neighbours — its aggregated vector must differ from its base.
    base_b = base["b"] / np.linalg.norm(base["b"])
    assert not np.allclose(agg["b"], base_b, atol=1e-4), (
        "aggregated vector for a connected node should differ from its base"
    )

    # All aggregated vectors are unit-normalised.
    for n in _NODES:
        assert np.isclose(np.linalg.norm(agg[n]), 1.0, atol=1e-5)

    # Isolated node 'f' falls back to the L2-normalised base vector exactly.
    expected_f = base["f"] / np.linalg.norm(base["f"])
    np.testing.assert_allclose(agg["f"], expected_f, atol=1e-6)


def test_adjacency_excludes_non_semantic_predicates(tmp_path):
    """_build_adjacency drops HAS_FILE_TYPE / CONTAINS_KEY / HAS_TAG edges."""
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE triples (subject TEXT, predicate TEXT, object TEXT)"
    )
    conn.executemany(
        "INSERT INTO triples(subject, predicate, object) VALUES (?,?,?)",
        [
            ("a", "RELATES_TO", "b"),    # semantic — kept
            ("a", "HAS_FILE_TYPE", "c"), # noise — dropped
            ("a", "CONTAINS_KEY", "d"),  # noise — dropped
            ("b", "HAS_TAG", "e"),       # noise — dropped
        ],
    )
    conn.commit()

    p = _pipeline(tmp_path)

    class _Store:
        _conn = conn

    p._store = _Store()
    adj = p._build_adjacency(["a", "b", "c", "d", "e"])

    assert adj["a"] == ["b"]
    assert adj["b"] == ["a"]
    assert adj["c"] == [] and adj["d"] == [] and adj["e"] == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
