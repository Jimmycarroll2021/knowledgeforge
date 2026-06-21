"""Entity-resolution F1 evaluation — roadmap gate >= 0.85.

Measures the CURRENT EntityResolver against a hand-labelled benchmark of
graph-ML-domain entity ids. A pair is PREDICTED same iff both ids resolve to
the same canonical id via EntityResolver.canonical(). Gold labels come from
tests/fixtures/er_labelled_pairs.json.

This only MEASURES the resolver — it does not modify resolution logic. If the
current resolver scores below the 0.85 gate the test is red by design; the
threshold is the roadmap gate and must not be weakened.
"""
from __future__ import annotations

import json
from pathlib import Path

from knowledgeforge.store.sqlite import SQLiteGraphStore
from knowledgeforge.resolution.resolver import EntityResolver
from knowledgeforge.embeddings.pipeline import EmbeddingPipeline
from knowledgeforge.contracts import now_iso

FIXTURE = Path(__file__).parent / "fixtures" / "er_labelled_pairs.json"


def _load_pairs() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _canonical_root(resolver: EntityResolver, entity_id: str) -> str:
    """Follow the alias chain to a fixed point.

    resolver.canonical() resolves one hop; chains can form when phases merge
    overlapping pairs. Iterating to a fixed point makes "same canonical cluster"
    detection correct without altering resolution itself.
    """
    seen: set[str] = set()
    current = entity_id
    while True:
        nxt = resolver.canonical(current)
        if nxt == current or nxt in seen:
            return nxt
        seen.add(current)
        current = nxt


def evaluate(resolver: EntityResolver, pairs: list[dict]) -> dict:
    """Compute precision/recall/F1 of the resolver against labelled pairs.

    Predicted-positive: both ids resolve to the same canonical root.
    Gold-positive: pair["same"] is True.
    """
    tp = fp = fn = 0
    for p in pairs:
        predicted_same = _canonical_root(resolver, p["id_a"]) == _canonical_root(
            resolver, p["id_b"]
        )
        gold_same = bool(p["same"])
        if predicted_same and gold_same:
            tp += 1
        elif predicted_same and not gold_same:
            fp += 1
        elif not predicted_same and gold_same:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _build_store_with_pairs(db_path: Path, pairs: list[dict]) -> SQLiteGraphStore:
    """Insert every id in the fixture as an entity of one common kind.

    All ids share a kind so the resolver's name-similarity logic (its actual
    job) decides merges, rather than getting a free pass from type-blocking —
    which would make the confusable negatives trivially correct.
    """
    store = SQLiteGraphStore(db_path)
    ids = sorted({i for p in pairs for i in (p["id_a"], p["id_b"])})
    ts = now_iso()
    with store._conn:
        for eid in ids:
            store._conn.execute(
                "INSERT OR IGNORE INTO entities(id, kind, created_at) VALUES (?,?,?)",
                (eid, "concept", ts),
            )
    return store


def test_resolution_f1_meets_gate(tmp_path):
    pairs = _load_pairs()
    store = _build_store_with_pairs(tmp_path / "er_eval.db", pairs)

    # Inject the embedding pipeline so the resolver's semantic-confirmation stage
    # is active: combined confidence = jaro*0.6 + cosine*0.4 plus a cosine floor
    # on non-identical pairs. This is what separates near-string confusables
    # (GraphSAGE/GraphSAINT, TransE/TransR) from true surface variants. embed_query
    # encodes on the fly, so no pre-build / ChromaDB write is needed.
    embed = EmbeddingPipeline(store, chroma_path=str(tmp_path / "emb"))

    resolver = EntityResolver(store, embed=embed)
    resolver.resolve()

    metrics = evaluate(resolver, pairs)

    print("\n=== Entity-Resolution F1 Evaluation (gate >= 0.85) ===")
    print(f"  pairs:     {len(pairs)} "
          f"(pos={sum(1 for p in pairs if p['same'])}, "
          f"neg={sum(1 for p in pairs if not p['same'])})")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall:    {metrics['recall']:.4f}")
    print(f"  f1:        {metrics['f1']:.4f}")
    print(f"  tp={metrics['tp']}  fp={metrics['fp']}  fn={metrics['fn']}")
    print("=" * 54)

    store.close()

    assert metrics["f1"] >= 0.85, (
        f"Entity-resolution F1 {metrics['f1']:.4f} below roadmap gate 0.85 "
        f"(precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
        f"tp={metrics['tp']}, fp={metrics['fp']}, fn={metrics['fn']})"
    )
