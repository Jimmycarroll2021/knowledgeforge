"""VaultAdapter — ingests Obsidian/markdown vaults into the KnowledgeForge graph.

Extracted and cleaned from redact-au knowledgegraph.py.
Swap include_dirs in AdapterSchema to target different vault layouts.
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

from ..contracts import AdapterSchema, SourceDocument, Triple, now_iso

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_WIKI_LINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]{1,160})\]\(([^)]+)\)")
_TAG_RE = re.compile(r"(?<![\w/])#([A-Za-z][A-Za-z0-9_-]{1,60})\b")

_DEFAULT_INCLUDE_DIRS: set[str] = {
    "algorithms", "books", "concepts", "knowledgeforge",
    "paper-notes", "papers", "people", "system",
}
_DEFAULT_EXCLUDE_DIRS: set[str] = {
    ".codex", ".codex-home-test", ".git", ".hg", ".obsidian",
    ".planning", ".ruff_cache", ".venv", "__pycache__",
    "logs", "manifests", "migration", "node_modules",
    "raw", "scripts", "tmp", "tools",
}
_DEFAULT_SUFFIXES: set[str] = {".md", ".txt", ".json", ".jsonl"}
_MAX_FILE_BYTES = 1_048_576


class VaultAdapter:
    """Adapter for Obsidian/markdown vaults.

    Pass include_dirs=None to ingest ALL non-excluded directories.
    Pass include_dirs={"algorithms", "concepts"} to restrict scope.
    """

    ADAPTER_NAME = "vault"

    def __init__(
        self,
        include_dirs: set[str] | None = None,
        exclude_dirs: set[str] | None = None,
        allowed_suffixes: set[str] | None = None,
        max_file_bytes: int = _MAX_FILE_BYTES,
        max_facts_per_file: int = 40,
    ) -> None:
        self._include_dirs = include_dirs  # None = all dirs
        self._exclude_dirs = exclude_dirs or _DEFAULT_EXCLUDE_DIRS
        self._suffixes = allowed_suffixes or _DEFAULT_SUFFIXES
        self._max_bytes = max_file_bytes
        self._max_facts = max_facts_per_file

    def schema(self) -> AdapterSchema:
        return AdapterSchema(
            adapter_name=self.ADAPTER_NAME,
            predicates=[
                "HAS_FILE_TYPE", "CONTAINS_HEADING", "LINKS_TO",
                "REFERENCES_URL_HOST", "HAS_TAG",
            ],
            source_kinds=["document"],
            target_kinds=["file_type", "heading:h1", "heading:h2", "heading:h3",
                          "vault_link", "local_link", "url_host", "tag"],
            include_dirs=sorted(self._include_dirs or []),
            allowed_suffixes=sorted(self._suffixes),
        )

    def scan(self, source: Path) -> list[SourceDocument]:
        root = source.expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Source path is not a directory: {root}")

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
            if self._include_dirs is not None and not self._include_allowed(rel):
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
        try:
            text = doc.path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        return self._extract_triples(doc, text)

    # ── private ───────────────────────────────────────────────────────────────

    def _excluded(self, rel: Path) -> bool:
        for part in rel.parts:
            if part.lower() in self._exclude_dirs or part.lower().startswith("tmp-"):
                return True
        return False

    def _include_allowed(self, rel: Path) -> bool:
        if len(rel.parts) == 1:
            return True
        return rel.parts[0].lower() in (self._include_dirs or set())

    def _extract_triples(self, doc: SourceDocument, text: str) -> list[Triple]:
        triples: list[Triple] = []
        seen: set[tuple[str, str, str]] = set()
        ts = now_iso()

        def add(
            predicate: str,
            obj: str,
            target_kind: str,
            evidence: str,
            confidence: float,
        ) -> None:
            if len(triples) >= self._max_facts:
                return
            subject = doc.doc_id
            key = (subject, predicate, obj)
            if key in seen or not obj.strip():
                return
            seen.add(key)
            triples.append(Triple(
                subject=subject,
                predicate=predicate,
                object=obj[:240],
                source_kind="document",
                target_kind=target_kind,
                evidence=evidence[:700],
                confidence=confidence,
                source_doc=doc.relative_path,
                extraction_method="rule",
                timestamp=ts,
                adapter=self.ADAPTER_NAME,
            ))

        add(
            "HAS_FILE_TYPE",
            doc.path.suffix.lower().lstrip(".") or "text",
            "file_type",
            f"File: {doc.relative_path}",
            0.95,
        )

        for line in text.splitlines():
            stripped = line.strip()
            if len(triples) >= self._max_facts:
                break
            if not stripped:
                continue

            m = _HEADING_RE.match(stripped)
            if m:
                level = len(m.group(1))
                add("CONTAINS_HEADING", m.group(2).strip(),
                    f"heading:h{level}", f"H{level}: {m.group(2).strip()}", 0.9)

            for m in _WIKI_LINK_RE.finditer(stripped):
                add("LINKS_TO", m.group(1).strip(), "vault_link", stripped, 0.86)

            for m in _MARKDOWN_LINK_RE.finditer(stripped):
                label, url = m.group(1).strip(), m.group(2).strip()
                parsed = urlparse(url)
                if parsed.scheme in {"http", "https"} and parsed.hostname:
                    add("REFERENCES_URL_HOST", parsed.hostname, "url_host",
                        f"Link: {label}", 0.82)
                elif url and not parsed.scheme:
                    add("LINKS_TO", label or url, "local_link", f"Link: {label}", 0.80)

            if not stripped.startswith("#"):
                for m in _TAG_RE.finditer(stripped):
                    add("HAS_TAG", m.group(1), "tag", stripped, 0.82)

        return triples
