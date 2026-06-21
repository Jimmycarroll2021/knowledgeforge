from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from ..models import IngestRequest, IngestResponse
from ...adapter.vault import VaultAdapter
from ...adapter.universal import UniversalAdapter
from ...pipeline import ForgePipeline

router = APIRouter()

_ADAPTERS = {"vault": VaultAdapter, "universal": UniversalAdapter}


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, request: Request) -> IngestResponse:
    source = Path(req.source)
    if not source.exists():
        raise HTTPException(status_code=400, detail=f"Source path does not exist: {req.source}")
    if req.adapter not in _ADAPTERS:
        raise HTTPException(status_code=400, detail=f"Unknown adapter: {req.adapter}")

    adp = _ADAPTERS[req.adapter](max_facts_per_file=req.max_facts)
    pipeline = ForgePipeline(request.app.state.store)
    result = pipeline.run(adp, source, dry_run=req.dry_run)

    return IngestResponse(
        adapter=result.adapter,
        source=result.source,
        documents_scanned=result.documents_scanned,
        documents_extracted=result.documents_extracted,
        triples_added=result.triples_added,
        triples_skipped=result.triples_skipped,
        errors=result.errors,
    )
