"""ForgePipeline — orchestrates scan → validate → extract → store for any Adapter.

Schema validation (Phase 7): checks predicate, source_kind, target_kind against
AdapterSchema, plus SHACL-style PropertyShape constraints (target class, datatype,
cardinality) before insert. Two modes:
  - soft (default): all triples admitted; Violations recorded as messages.
  - strict (opt-in): Violation-severity triples are rejected (excluded + counted);
    Warning/Info are always admit-and-flag.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .contracts import Adapter, AdapterSchema, IngestResult, PropertyShape, Triple

if TYPE_CHECKING:
    from .store.sqlite import SQLiteGraphStore


class ForgePipeline:
    def __init__(self, store: SQLiteGraphStore) -> None:
        self._store = store

    def run(
        self,
        adapter: Adapter,
        source: Path,
        dry_run: bool = False,
        strict: bool = False,
    ) -> IngestResult:
        docs = adapter.scan(source)
        schema = adapter.schema()

        if dry_run:
            return IngestResult(
                adapter=schema.adapter_name,
                source=str(source),
                documents_scanned=len(docs),
                documents_extracted=0,
                triples_added=0,
                triples_skipped=0,
            )

        total_added = total_skipped = total_rejected = 0
        extracted_docs = 0
        errors: list[str] = []

        for doc in docs:
            try:
                triples = adapter.extract(doc)
                validated, violations, rejected = self._validate(triples, schema, strict)
                errors.extend(violations)
                total_rejected += rejected
                added, skipped = self._store.add_triples(validated)
                total_added += added
                total_skipped += skipped
                if validated:
                    extracted_docs += 1
            except Exception as exc:
                errors.append(f"{doc.relative_path}: {exc}")

        return IngestResult(
            adapter=schema.adapter_name,
            source=str(source),
            documents_scanned=len(docs),
            documents_extracted=extracted_docs,
            triples_added=total_added,
            triples_skipped=total_skipped,
            triples_rejected=total_rejected,
            errors=errors,
        )

    def _validate(
        self,
        triples: list[Triple],
        schema: AdapterSchema,
        strict: bool,
    ) -> tuple[list[Triple], list[str], int]:
        """SHACL-style validation against the AdapterSchema.

        Severity ladder per triple:
          - unknown predicate / source_kind / target_kind → Violation
          - PropertyShape match: target_kind not in allowed_target_kinds → Violation;
            datatype mismatch → shape.severity; cardinality min/max per subject →
            shape.severity
        In strict mode a triple carrying any Violation is excluded and counted as
        rejected. In soft mode everything is admitted; all findings are recorded as
        messages regardless of severity.

        Returns (valid_triples, violation_messages, rejected_count).
        """
        valid_predicates = set(schema.predicates) if schema.predicates else None
        valid_source_kinds = set(schema.source_kinds) if schema.source_kinds else None
        valid_target_kinds = set(schema.target_kinds) if schema.target_kinds else None
        shapes_by_predicate: dict[str, list[PropertyShape]] = {}
        for shape in schema.shapes:
            shapes_by_predicate.setdefault(shape.predicate, []).append(shape)

        # Cardinality is per (subject, predicate) within this document's triples.
        pred_counts: dict[tuple[str, str], int] = {}
        for t in triples:
            pred_counts[(t.subject, t.predicate)] = pred_counts.get((t.subject, t.predicate), 0) + 1

        valid: list[Triple] = []
        messages: list[str] = []
        rejected = 0

        for t in triples:
            findings: list[tuple[str, str]] = []  # (severity, detail)

            if valid_predicates and t.predicate not in valid_predicates:
                findings.append(("Violation", f"unknown predicate '{t.predicate}'"))
            if valid_source_kinds and t.source_kind not in valid_source_kinds:
                findings.append(("Violation", f"unknown source_kind '{t.source_kind}'"))
            if valid_target_kinds and t.target_kind not in valid_target_kinds:
                findings.append(("Violation", f"unknown target_kind '{t.target_kind}'"))

            for shape in shapes_by_predicate.get(t.predicate, ()):
                if shape.allowed_target_kinds and t.target_kind not in shape.allowed_target_kinds:
                    findings.append((
                        "Violation",
                        f"target_kind '{t.target_kind}' not in allowed {shape.allowed_target_kinds}",
                    ))
                if (
                    shape.datatype is not None
                    and t.object_is_literal
                    and t.object_datatype != shape.datatype
                ):
                    findings.append((
                        shape.severity,
                        f"datatype '{t.object_datatype}' != required '{shape.datatype}'",
                    ))
                count = pred_counts[(t.subject, t.predicate)]
                if count < shape.min_count:
                    findings.append((
                        shape.severity,
                        f"cardinality {count} < min_count {shape.min_count}",
                    ))
                if shape.max_count is not None and count > shape.max_count:
                    findings.append((
                        shape.severity,
                        f"cardinality {count} > max_count {shape.max_count}",
                    ))

            if findings:
                top = "Violation" if any(s == "Violation" for s, _ in findings) else findings[0][0]
                detail = "; ".join(f"[{s}] {d}" for s, d in findings)
                messages.append(
                    f"SCHEMA {top.upper()} ({t.subject}→{t.predicate}→{t.object}): {detail}"
                )
                if strict and any(s == "Violation" for s, _ in findings):
                    rejected += 1
                    continue

            valid.append(t)

        return valid, messages, rejected
