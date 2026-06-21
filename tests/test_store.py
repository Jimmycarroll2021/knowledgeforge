from knowledgeforge.store.sqlite import SQLiteGraphStore
from knowledgeforge.contracts import Triple


def make_triple(subj="doc:abc", pred="LINKS_TO", obj="GraphSAGE", source_doc="notes.md") -> Triple:
    return Triple(
        subject=subj, predicate=pred, object=obj,
        source_kind="document", target_kind="vault_link",
        evidence="[[GraphSAGE]]", confidence=0.86,
        source_doc=source_doc, extraction_method="rule",
        timestamp="2026-06-21T00:00:00+00:00", adapter="vault",
    )


def test_add_triple(tmp_path):
    store = SQLiteGraphStore(tmp_path / "test.db")
    t = make_triple()
    assert store.add_triple(t) is True
    assert store.add_triple(t) is False  # duplicate
    store.close()


def test_stats(tmp_path):
    store = SQLiteGraphStore(tmp_path / "test.db")
    store.add_triple(make_triple())
    store.add_triple(make_triple(pred="HAS_TAG", obj="graph-ml"))
    s = store.stats()
    assert s["triples"] == 2
    assert s["entities"] >= 2
    store.close()


def test_query_by_predicate(tmp_path):
    store = SQLiteGraphStore(tmp_path / "test.db")
    store.add_triple(make_triple(pred="LINKS_TO"))
    store.add_triple(make_triple(pred="HAS_TAG", obj="graph-ml"))
    rows = store.query(predicate="LINKS_TO")
    assert len(rows) == 1
    assert rows[0]["predicate"] == "LINKS_TO"
    store.close()


def test_provenance(tmp_path):
    store = SQLiteGraphStore(tmp_path / "test.db")
    t = make_triple()
    store.add_triple(t)
    rows = store.provenance(t.subject)
    assert len(rows) >= 1
    assert rows[0]["subject"] == t.subject
    store.close()


def test_bulk_add(tmp_path):
    store = SQLiteGraphStore(tmp_path / "test.db")
    triples = [make_triple(obj=f"Entity{i}", source_doc=f"doc{i}.md") for i in range(20)]
    added, skipped = store.add_triples(triples)
    assert added == 20
    assert skipped == 0
    added2, skipped2 = store.add_triples(triples)
    assert added2 == 0
    assert skipped2 == 20
    store.close()
