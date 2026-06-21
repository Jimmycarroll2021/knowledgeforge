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
    # ── PROV-O provenance dimensions (W3C PROV-O alignment) ──────────────────
    source: str = ""  # prov origin/agent — the producing process or person, distinct from source_doc (the file path)
    lineage: tuple[str, ...] = ()  # prov:wasDerivedFrom — parent triple_ids; empty for raw source facts
    valid_from: str | None = None  # bitemporal validity start (ISO-8601)
    valid_to: str | None = None  # bitemporal validity end (None = still valid)
    schema_version: str = "1"  # contract version this triple was minted under
    # ── RDF literal-vs-entity distinction ───────────────────────────────────
    object_is_literal: bool = False  # True when object is a literal value, not an entity node
    object_datatype: str | None = None  # xsd-style datatype for literal objects (e.g. "xsd:integer")

    def triple_id(self) -> str:
        # layer is part of identity so the same (s,p,o) fact can coexist across
        # derivation tiers (a source fact and an inferred relation are distinct).
        key = f"{self.subject}|{self.predicate}|{self.object}|{self.source_doc}|{self.layer}"
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
class PropertyShape:
    """SHACL-style property shape — constrains the triples for one predicate.

    Drives real shape validation in ForgePipeline (cardinality, target class,
    datatype) rather than mere predicate-set membership. severity follows
    SHACL's sh:severity (Violation rejects, Warning/Info admit-and-flag).
    """
    predicate: str
    allowed_target_kinds: list[str] = field(default_factory=list)  # empty = any kind allowed
    min_count: int = 0  # min occurrences per subject (0 = optional)
    max_count: int | None = None  # max occurrences per subject (None = unbounded)
    datatype: str | None = None  # required xsd datatype when the object is a literal
    severity: str = "Violation"  # Violation | Warning | Info


@dataclass
class AdapterSchema:
    adapter_name: str
    predicates: list[str]
    source_kinds: list[str]
    target_kinds: list[str]
    include_dirs: list[str] = field(default_factory=list)
    allowed_suffixes: list[str] = field(default_factory=lambda: [".md", ".txt"])
    shapes: list[PropertyShape] = field(default_factory=list)  # SHACL-style node/property shapes


@dataclass
class IngestResult:
    adapter: str
    source: str
    documents_scanned: int
    documents_extracted: int
    triples_added: int
    triples_skipped: int
    triples_rejected: int = 0  # excluded by hard (strict) schema validation
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Adapter:    {self.adapter}",
            f"Source:     {self.source}",
            f"Scanned:    {self.documents_scanned} documents",
            f"Extracted:  {self.documents_extracted} documents",
            f"Triples:    {self.triples_added} added, {self.triples_skipped} skipped, {self.triples_rejected} rejected",
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
