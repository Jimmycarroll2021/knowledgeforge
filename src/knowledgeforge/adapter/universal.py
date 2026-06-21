"""UniversalAdapter — ingests any file type into the KnowledgeForge graph.

Drop any folder of files — markdown, PDF, Word, HTML, CSV, JSON, code,
plain text — and get a knowledge graph with provenance on every triple.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

from ..contracts import Adapter, AdapterSchema, SourceDocument, Triple, now_iso
from .extract import extract_text

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_WIKI_LINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]{1,160})\]\(([^)]+)\)")
_TAG_RE = re.compile(r"(?<![\w/])#([A-Za-z][A-Za-z0-9_-]{1,60})\b")
_URL_RE = re.compile(r"https?://([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})(?:[/?#][^\s\"'<>]*)?")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

# File types we can handle
_TEXT_LIKE = {
    ".md", ".txt", ".rst", ".org", ".tex",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".rb", ".java",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".php", ".sh", ".bash", ".zsh",
    ".toml", ".ini", ".cfg", ".conf", ".env",
    ".xml", ".svg",
}
_STRUCTURED = {".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml"}
_RICH_DOC = {".pdf", ".docx", ".doc", ".html", ".htm"}

_SUPPORTED_SUFFIXES = _TEXT_LIKE | _STRUCTURED | _RICH_DOC

_ALWAYS_EXCLUDE = {
    ".git", ".hg", ".venv", "__pycache__", ".ruff_cache", ".mypy_cache",
    ".pytest_cache", "node_modules", "dist", "build", ".eggs",
}

_MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB


class UniversalAdapter:
    """Ingests any file type. No domain config needed — drop any folder."""

    ADAPTER_NAME = "universal"

    def __init__(
        self,
        exclude_dirs: set[str] | None = None,
        extra_suffixes: set[str] | None = None,
        max_file_bytes: int = _MAX_FILE_BYTES,
        max_facts_per_file: int = 60,
    ) -> None:
        self._exclude_dirs = (exclude_dirs or set()) | _ALWAYS_EXCLUDE
        self._suffixes = _SUPPORTED_SUFFIXES | (extra_suffixes or set())
        self._max_bytes = max_file_bytes
        self._max_facts = max_facts_per_file

    def schema(self) -> AdapterSchema:
        return AdapterSchema(
            adapter_name=self.ADAPTER_NAME,
            predicates=[
                "HAS_FILE_TYPE", "CONTAINS_HEADING", "LINKS_TO",
                "REFERENCES_URL", "HAS_TAG", "CONTAINS_KEY",
                "HAS_COLUMN", "MENTIONS",
            ],
            source_kinds=["document"],
            target_kinds=[
                "file_type", "heading", "vault_link", "url_host",
                "tag", "key", "column", "entity",
            ],
            allowed_suffixes=sorted(self._suffixes),
        )

    def scan(self, source: Path) -> list[SourceDocument]:
        root = source.expanduser().resolve()
        if root.is_file():
            # single file mode
            doc = SourceDocument.from_path(root, root.parent)
            return [doc] if root.stat().st_size <= self._max_bytes else []
        if not root.is_dir():
            raise ValueError(f"Source must be a file or directory: {root}")

        docs: list[SourceDocument] = []
        for path in sorted(root.rglob("*"), key=lambda p: str(p).lower()):
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(root)
            except ValueError:
                continue
            if self._excluded(rel):
                continue
            if path.suffix.lower() not in self._suffixes:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size <= 0 or size > self._max_bytes:
                continue
            docs.append(SourceDocument.from_path(path, root))
        return docs

    def extract(self, doc: SourceDocument) -> list[Triple]:
        text = extract_text(doc.path)
        if not text or not text.strip():
            return []
        suffix = doc.path.suffix.lower()
        triples: list[Triple] = []

        if suffix in _STRUCTURED:
            triples.extend(self._extract_structured(doc, text, suffix))
        else:
            triples.extend(self._extract_text(doc, text))

        # always add file type triple
        file_type_triple = self._make(
            doc, "HAS_FILE_TYPE", suffix.lstrip(".") or "text",
            "file_type", f"File: {doc.relative_path}", 0.95,
        )
        if file_type_triple and not any(t.predicate == "HAS_FILE_TYPE" for t in triples):
            triples.insert(0, file_type_triple)

        return triples[: self._max_facts]

    # ── extractors ────────────────────────────────────────────────────────────

    def _extract_text(self, doc: SourceDocument, text: str) -> list[Triple]:
        triples: list[Triple] = []
        seen: set[tuple[str, str, str]] = set()
        ts = now_iso()

        def add(predicate: str, obj: str, kind: str, evidence: str, conf: float) -> None:
            if len(triples) >= self._max_facts:
                return
            key = (doc.doc_id, predicate, obj[:240])
            if key in seen or not obj.strip():
                return
            seen.add(key)
            triples.append(Triple(
                subject=doc.doc_id,
                predicate=predicate,
                object=obj[:240],
                source_kind="document",
                target_kind=kind,
                evidence=evidence[:700],
                confidence=conf,
                source_doc=doc.relative_path,
                extraction_method="rule",
                timestamp=ts,
                adapter=self.ADAPTER_NAME,
            ))

        for line in text.splitlines():
            stripped = line.strip()
            if len(triples) >= self._max_facts:
                break
            if not stripped:
                continue

            # headings (markdown or ALL-CAPS lines as section markers)
            m = _HEADING_RE.match(stripped)
            if m:
                level = len(m.group(1))
                add("CONTAINS_HEADING", m.group(2).strip(),
                    f"heading:h{level}", f"H{level}: {m.group(2).strip()}", 0.90)
            elif len(stripped) < 80 and stripped.isupper() and len(stripped.split()) <= 8:
                add("CONTAINS_HEADING", stripped.title(), "heading", stripped, 0.70)

            # wiki links
            for m in _WIKI_LINK_RE.finditer(stripped):
                add("LINKS_TO", m.group(1).strip(), "vault_link", stripped, 0.86)

            # markdown links / URLs
            for m in _MARKDOWN_LINK_RE.finditer(stripped):
                label, url = m.group(1).strip(), m.group(2).strip()
                parsed = urlparse(url)
                if parsed.scheme in {"http", "https"} and parsed.hostname:
                    add("REFERENCES_URL", parsed.hostname, "url_host",
                        f"Link: {label}", 0.82)

            # bare URLs
            for m in _URL_RE.finditer(stripped):
                add("REFERENCES_URL", m.group(1), "url_host", stripped, 0.75)

            # hashtags (not in headings)
            if not stripped.startswith("#"):
                for m in _TAG_RE.finditer(stripped):
                    add("HAS_TAG", m.group(1), "tag", stripped, 0.82)

        return triples

    def _extract_structured(
        self, doc: SourceDocument, text: str, suffix: str
    ) -> list[Triple]:
        """Extract typed triples from structured files (CSV, JSON, YAML)."""
        triples: list[Triple] = []
        seen: set[tuple[str, str, str]] = set()
        ts = now_iso()

        def add(predicate: str, obj: str, kind: str, evidence: str, conf: float) -> None:
            if len(triples) >= self._max_facts:
                return
            key = (doc.doc_id, predicate, obj[:240])
            if key in seen or not obj.strip():
                return
            seen.add(key)
            triples.append(Triple(
                subject=doc.doc_id,
                predicate=predicate,
                object=obj[:240],
                source_kind="document",
                target_kind=kind,
                evidence=evidence[:700],
                confidence=conf,
                source_doc=doc.relative_path,
                extraction_method="structural",
                timestamp=ts,
                adapter=self.ADAPTER_NAME,
            ))

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or len(triples) >= self._max_facts:
                break

            if stripped.startswith("Columns:"):
                cols = stripped[len("Columns:"):].strip()
                for col in cols.split(","):
                    col = col.strip()
                    if col:
                        add("HAS_COLUMN", col, "column", stripped, 0.90)
                continue

            # key: value pairs from JSON/YAML flattening
            if ": " in stripped and not stripped.startswith("-"):
                parts = stripped.split(": ", 1)
                key = parts[0].strip().lstrip(".")
                value = parts[1].strip()
                if key and value and len(value) < 200:
                    add("CONTAINS_KEY", f"{key}={value}", "key",
                        stripped, 0.80)

        # also run text extraction to catch any narrative content
        text_triples = self._extract_text(doc, text)
        for t in text_triples:
            key = (t.subject, t.predicate, t.object)
            if key not in seen and len(triples) < self._max_facts:
                seen.add(key)
                triples.append(t)

        return triples

    # ── helpers ───────────────────────────────────────────────────────────────

    def _excluded(self, rel: Path) -> bool:
        for part in rel.parts:
            if part.lower() in self._exclude_dirs or part.lower().startswith("."):
                return True
        return False

    def _make(
        self,
        doc: SourceDocument,
        predicate: str,
        obj: str,
        kind: str,
        evidence: str,
        conf: float,
    ) -> Triple | None:
        if not obj.strip():
            return None
        return Triple(
            subject=doc.doc_id,
            predicate=predicate,
            object=obj[:240],
            source_kind="document",
            target_kind=kind,
            evidence=evidence[:700],
            confidence=conf,
            source_doc=doc.relative_path,
            extraction_method="rule",
            timestamp=now_iso(),
            adapter=self.ADAPTER_NAME,
        )
