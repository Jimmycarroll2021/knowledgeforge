"""Tests for the text extractor layer."""
import json
from knowledgeforge.adapter.extract import extract_text, _flatten_json


def test_extract_markdown(tmp_path):
    f = tmp_path / "note.md"
    f.write_text("# Hello\n\nSome text with [[links]].")
    result = extract_text(f)
    assert "Hello" in result
    assert "links" in result


def test_extract_txt(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("Plain text content here.")
    assert "Plain text" in extract_text(f)


def test_extract_python(tmp_path):
    f = tmp_path / "script.py"
    f.write_text("def hello():\n    return 'world'\n")
    assert "hello" in extract_text(f)


def test_extract_json_flat(tmp_path):
    f = tmp_path / "data.json"
    f.write_text(json.dumps({"title": "GraphSAGE", "year": 2017}))
    result = extract_text(f)
    assert "GraphSAGE" in result
    assert "2017" in result


def test_extract_jsonl(tmp_path):
    f = tmp_path / "data.jsonl"
    lines = [json.dumps({"id": i, "name": f"entity{i}"}) for i in range(5)]
    f.write_text("\n".join(lines))
    result = extract_text(f)
    assert "entity0" in result


def test_extract_csv(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("name,category\nfoo,A\nbar,B\n")
    result = extract_text(f)
    assert "Columns:" in result
    assert "name" in result
    assert "foo" in result


def test_extract_missing_file(tmp_path):
    result = extract_text(tmp_path / "missing.pdf")
    assert result == ""


def test_flatten_json_nested():
    obj = {"a": {"b": {"c": "deep"}}, "x": [1, 2]}
    result = _flatten_json(obj)
    assert "a.b.c: deep" in result


def test_html_extraction(tmp_path):
    f = tmp_path / "page.html"
    f.write_text(
        "<html><body>"
        "<script>var x=1;</script>"
        "<h1>Title</h1><p>Content here.</p>"
        "</body></html>"
    )
    result = extract_text(f)
    assert "Title" in result
    assert "Content" in result
    assert "var x" not in result  # script stripped
