"""ForgePipeline — orchestrates scan → extract → store for any Adapter."""
from __future__ import annotations

from pathlib import Path

from .contracts import Adapter, IngestResult
from .store.sqlite import SQLiteGraphStore


class ForgePipeline:
    def __init__(self, store: SQLiteGraphStore) -> None:
        self._store = store

    def run(self, adapter: Adapter, source: Path, dry_run: bool = False) -> IngestResult:
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

        total_added = total_skipped = 0
        extracted_docs = 0
        errors: list[str] = []

        for doc in docs:
            try:
                triples = adapter.extract(doc)
                added, skipped = self._store.add_triples(triples)
                total_added += added
                total_skipped += skipped
                if triples:
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
            errors=errors,
        )
