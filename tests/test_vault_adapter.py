from knowledgeforge.adapter.vault import VaultAdapter
from knowledgeforge.contracts import Adapter


def test_adapter_protocol():
    adp = VaultAdapter()
    assert isinstance(adp, Adapter)


def test_schema(tmp_path):
    adp = VaultAdapter()
    schema = adp.schema()
    assert schema.adapter_name == "vault"
    assert "LINKS_TO" in schema.predicates
    assert "HAS_TAG" in schema.predicates


def test_scan_finds_markdown(tmp_path):
    (tmp_path / "graphsage.md").write_text("# GraphSAGE\n[[Entity Resolution]]")
    (tmp_path / "notes.txt").write_text("some text")
    (tmp_path / "skip.bin").write_bytes(b"\x00\x01")

    adp = VaultAdapter()
    docs = adp.scan(tmp_path)
    names = [d.path.name for d in docs]
    assert "graphsage.md" in names
    assert "notes.txt" in names
    assert "skip.bin" not in names


def test_scan_excludes_venv(tmp_path):
    venv = tmp_path / ".venv"
    venv.mkdir()
    (venv / "lib.md").write_text("# lib")
    (tmp_path / "real.md").write_text("# real")

    adp = VaultAdapter()
    docs = adp.scan(tmp_path)
    paths = [d.relative_path for d in docs]
    assert not any(".venv" in p for p in paths)
    assert "real.md" in paths


def test_extract_triples(tmp_path):
    f = tmp_path / "graphsage.md"
    f.write_text(
        "# GraphSAGE\n\n"
        "tags: #graph-ml\n\n"
        "See [[Entity Resolution]] and [[TransE]].\n\n"
        "[Paper](https://arxiv.org/abs/1706.02216)\n"
    )
    from knowledgeforge.contracts import SourceDocument
    doc = SourceDocument.from_path(f, tmp_path)
    adp = VaultAdapter()
    triples = adp.extract(doc)

    predicates = {t.predicate for t in triples}
    assert "HAS_FILE_TYPE" in predicates
    assert "CONTAINS_HEADING" in predicates
    assert "LINKS_TO" in predicates
    assert "HAS_TAG" in predicates
    assert "REFERENCES_URL_HOST" in predicates

    assert all(t.evidence for t in triples)
    assert all(0 < t.confidence <= 1.0 for t in triples)
    assert all(t.extraction_method == "rule" for t in triples)


def test_extract_no_duplicates(tmp_path):
    f = tmp_path / "dup.md"
    f.write_text("[[GraphSAGE]] and again [[GraphSAGE]]\n")
    from knowledgeforge.contracts import SourceDocument
    doc = SourceDocument.from_path(f, tmp_path)
    adp = VaultAdapter()
    triples = adp.extract(doc)
    links = [t for t in triples if t.predicate == "LINKS_TO" and t.object == "GraphSAGE"]
    assert len(links) == 1


def test_scan_include_dirs_filter(tmp_path):
    (tmp_path / "algorithms").mkdir()
    (tmp_path / "algorithms" / "graphsage.md").write_text("# GS")
    (tmp_path / "misc").mkdir()
    (tmp_path / "misc" / "other.md").write_text("# other")

    adp = VaultAdapter(include_dirs={"algorithms"})
    docs = adp.scan(tmp_path)
    paths = [d.relative_path for d in docs]
    assert any("algorithms" in p for p in paths)
    assert not any("misc" in p for p in paths)
