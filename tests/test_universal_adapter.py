import json
from knowledgeforge.adapter.universal import UniversalAdapter
from knowledgeforge.contracts import Adapter


def test_protocol():
    assert isinstance(UniversalAdapter(), Adapter)


def test_scan_any_file(tmp_path):
    (tmp_path / "notes.md").write_text("# Hello")
    (tmp_path / "data.csv").write_text("name,value\nfoo,1")
    (tmp_path / "config.json").write_text('{"key": "value"}')
    (tmp_path / "script.py").write_text("def main(): pass")
    (tmp_path / "readme.txt").write_text("Some text")
    (tmp_path / "page.html").write_text("<h1>Hello</h1>")

    adp = UniversalAdapter()
    docs = adp.scan(tmp_path)
    names = {d.path.name for d in docs}
    assert "notes.md" in names
    assert "data.csv" in names
    assert "config.json" in names
    assert "script.py" in names
    assert "readme.txt" in names
    assert "page.html" in names


def test_scan_excludes_venv(tmp_path):
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "lib.py").write_text("# lib")
    (tmp_path / "real.py").write_text("# real")

    adp = UniversalAdapter()
    docs = adp.scan(tmp_path)
    paths = [d.relative_path for d in docs]
    assert not any(".venv" in p for p in paths)
    assert "real.py" in paths


def test_scan_single_file(tmp_path):
    f = tmp_path / "single.txt"
    f.write_text("Hello world")
    adp = UniversalAdapter()
    docs = adp.scan(f)
    assert len(docs) == 1
    assert docs[0].path.name == "single.txt"


def test_extract_markdown(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("# GraphSAGE\n\ntags: #gnn\n\nSee [[Entity Resolution]].\n")
    from knowledgeforge.contracts import SourceDocument
    doc = SourceDocument.from_path(f, tmp_path)
    adp = UniversalAdapter()
    triples = adp.extract(doc)
    predicates = {t.predicate for t in triples}
    assert "CONTAINS_HEADING" in predicates
    assert "HAS_TAG" in predicates
    assert "LINKS_TO" in predicates


def test_extract_csv(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("name,category,value\nalpha,A,1\nbeta,B,2\n")
    from knowledgeforge.contracts import SourceDocument
    doc = SourceDocument.from_path(f, tmp_path)
    adp = UniversalAdapter()
    triples = adp.extract(doc)
    predicates = {t.predicate for t in triples}
    assert "HAS_COLUMN" in predicates
    cols = {t.object for t in triples if t.predicate == "HAS_COLUMN"}
    assert "name" in cols
    assert "category" in cols


def test_extract_json(tmp_path):
    f = tmp_path / "config.json"
    f.write_text(json.dumps({"title": "GraphSAGE", "year": 2017, "authors": ["Hamilton"]}))
    from knowledgeforge.contracts import SourceDocument
    doc = SourceDocument.from_path(f, tmp_path)
    adp = UniversalAdapter()
    triples = adp.extract(doc)
    assert len(triples) > 0
    assert all(t.source_doc == "config.json" for t in triples)


def test_extract_python(tmp_path):
    f = tmp_path / "model.py"
    f.write_text(
        "# GraphSAGE implementation\n"
        "import torch\n"
        "class GraphSAGE:\n"
        "    # see https://arxiv.org/abs/1706.02216\n"
        "    pass\n"
    )
    from knowledgeforge.contracts import SourceDocument
    doc = SourceDocument.from_path(f, tmp_path)
    adp = UniversalAdapter()
    triples = adp.extract(doc)
    assert any(t.predicate == "HAS_FILE_TYPE" and t.object == "py" for t in triples)
    assert any(t.predicate == "REFERENCES_URL" for t in triples)


def test_extract_html(tmp_path):
    f = tmp_path / "page.html"
    f.write_text(
        "<html><body>"
        "<h1>GraphSAGE</h1>"
        "<p>See <a href='https://arxiv.org'>arxiv.org</a></p>"
        "</body></html>"
    )
    from knowledgeforge.contracts import SourceDocument
    doc = SourceDocument.from_path(f, tmp_path)
    adp = UniversalAdapter()
    triples = adp.extract(doc)
    assert any(t.predicate == "CONTAINS_HEADING" for t in triples)
    assert any(t.predicate == "REFERENCES_URL" for t in triples)


def test_all_triples_have_provenance(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("# Test\n\n[[Link]] and #tag and https://example.com\n")
    from knowledgeforge.contracts import SourceDocument
    doc = SourceDocument.from_path(f, tmp_path)
    adp = UniversalAdapter()
    triples = adp.extract(doc)
    for t in triples:
        assert t.source_doc
        assert t.extraction_method in {"rule", "structural"}
        assert t.timestamp
        assert t.adapter == "universal"
        assert 0 < t.confidence <= 1.0
        assert t.evidence
