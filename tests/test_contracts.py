from knowledgeforge.contracts import Triple, SourceDocument, now_iso
from pathlib import Path
import hashlib


def make_triple(**kwargs) -> Triple:
    defaults = dict(
        subject="doc:abc123",
        predicate="LINKS_TO",
        object="GraphSAGE",
        source_kind="document",
        target_kind="vault_link",
        evidence="see [[GraphSAGE]]",
        confidence=0.86,
        source_doc="algorithms/graphsage.md",
        extraction_method="rule",
        timestamp="2026-06-21T00:00:00+00:00",
        adapter="vault",
    )
    defaults.update(kwargs)
    return Triple(**defaults)


def test_triple_id_stable():
    t = make_triple()
    assert t.triple_id() == t.triple_id()


def test_triple_id_differs_by_predicate():
    t1 = make_triple(predicate="LINKS_TO")
    t2 = make_triple(predicate="HAS_TAG")
    assert t1.triple_id() != t2.triple_id()


def test_source_document_from_path(tmp_path):
    f = tmp_path / "notes" / "graphsage.md"
    f.parent.mkdir()
    f.write_text("# GraphSAGE")
    doc = SourceDocument.from_path(f, tmp_path)
    assert doc.relative_path == "notes/graphsage.md"
    assert doc.doc_id.startswith("doc:")
    assert doc.size_bytes > 0


def test_now_iso_format():
    ts = now_iso()
    assert "T" in ts and "+" in ts or "Z" in ts
