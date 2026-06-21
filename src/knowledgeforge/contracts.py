from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Triple:
    subject: str
    predicate: str
    object: str
    source_kind: str
    target_kind: str
    evidence: str
    confidence: float
    source_doc: str
    extraction_method: str  # "rule" | "llm" | "structural"
    timestamp: str
    adapter: str
    layer: str = "source_facts"  # source_facts | normalised_triples | inferred_relations | llm_hypotheses

    def triple_id(self) -> str:
        key = f"{self.subject}|{self.predicate}|{self.object}|{self.source_doc}"
        return hashlib.sha256(key.encode()).hexdigest()[:24]


@dataclass(frozen=True)
class SourceDocument:
    path: Path
    relative_path: str
    size_bytes: int
    doc_id: str  # stable identifier for provenance

    @classmethod
    def from_path(cls, path: Path, root: Path) -> SourceDocument:
        rel = path.relative_to(root).as_posix()
        doc_id = "doc:" + hashlib.sha256(rel.encode()).hexdigest()[:16]
        return cls(path=path, relative_path=rel, size_bytes=path.stat().st_size, doc_id=doc_id)


@dataclass
class AdapterSchema:
    adapter_name: str
    predicates: list[str]
    source_kinds: list[str]
    target_kinds: list[str]
    include_dirs: list[str] = field(default_factory=list)
    allowed_suffixes: list[str] = field(default_factory=lambda: [".md", ".txt"])


@dataclass
class IngestResult:
    adapter: str
    source: str
    documents_scanned: int
    documents_extracted: int
    triples_added: int
    triples_skipped: int
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Adapter:    {self.adapter}",
            f"Source:     {self.source}",
            f"Scanned:    {self.documents_scanned} documents",
            f"Extracted:  {self.documents_extracted} documents",
            f"Triples:    {self.triples_added} added, {self.triples_skipped} skipped",
        ]
        if self.errors:
            lines.append(f"Errors:     {len(self.errors)}")
            for e in self.errors[:5]:
                lines.append(f"  - {e}")
        return "\n".join(lines)


@runtime_checkable
class Adapter(Protocol):
    def scan(self, source: Path) -> list[SourceDocument]: ...
    def extract(self, doc: SourceDocument) -> list[Triple]: ...
    def schema(self) -> AdapterSchema: ...


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
