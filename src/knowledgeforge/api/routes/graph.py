from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from ..models import (
    GraphStatsResponse,
    NodeResponse,
    PathEdge,
    PathResponse,
    ProvenanceResponse,
    TripleOut,
)
from ...inference.graphrag import GraphRAG
from ...community.detector import CommunityDetector
from ...resolution.resolver import EntityResolver

router = APIRouter(prefix="/graph")


def _triple_out(row: dict[str, Any]) -> TripleOut:
    return TripleOut(
        triple_id=row.get("triple_id", ""),
        subject=row["subject"],
        predicate=row["predicate"],
        object=row["object"],
        confidence=row["confidence"],
        source_doc=row.get("source_doc", ""),
        extraction_method=row.get("extraction_method", ""),
        layer=row.get("layer", "source_facts"),
        timestamp=row.get("timestamp", ""),
        evidence=row.get("evidence", ""),
    )


@router.get("/stats", response_model=GraphStatsResponse)
def graph_stats(request: Request) -> GraphStatsResponse:
    store = request.app.state.store
    s = store.stats()
    return GraphStatsResponse(
        entities=s["entities"],
        triples=s["triples"],
        by_layer=s.get("by_layer", {}),
        top_predicates=dict(list(s.get("by_predicate", {}).items())[:20]),
    )


@router.get("/node/{entity_id:path}", response_model=NodeResponse)
def node(entity_id: str, request: Request, limit: int = 50) -> NodeResponse:
    store = request.app.state.store
    row = store._conn.execute(
        "SELECT id, kind FROM entities WHERE id = ?", (entity_id,)
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Entity not found: {entity_id}")

    triples = store.query(subject=entity_id, limit=limit)
    triples += store.query(obj=entity_id, limit=limit)

    return NodeResponse(
        entity_id=row["id"],
        kind=row["kind"],
        triples=[_triple_out(t) for t in triples],
    )


@router.get("/provenance/{entity_id:path}", response_model=ProvenanceResponse)
def provenance(entity_id: str, request: Request) -> ProvenanceResponse:
    store = request.app.state.store
    rows = store.provenance(entity_id)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No provenance found for: {entity_id}")
    return ProvenanceResponse(
        entity=entity_id,
        triples=[_triple_out(r) for r in rows],
    )


@router.get("/path", response_model=PathResponse)
def path(
    request: Request,
    from_entity: str = Query(..., alias="from"),
    to_entity: str = Query(..., alias="to"),
    model: str = "claude-haiku-4-5-20251001",
) -> PathResponse:
    store = request.app.state.store
    rag = GraphRAG(store, model=model)
    paths = rag.path(from_entity, to_entity)
    if not paths:
        return PathResponse(
            from_entity=from_entity,
            to_entity=to_entity,
            path=[],
            found=False,
        )
    return PathResponse(
        from_entity=from_entity,
        to_entity=to_entity,
        path=[PathEdge(subject=e["subject"], predicate=e["predicate"], object=e["object"]) for e in paths[0]],
        found=True,
    )


@router.post("/community")
def detect_communities(request: Request) -> dict[str, Any]:
    store = request.app.state.store
    return CommunityDetector(store).detect_and_summarise()


@router.get("/community")
def communities(request: Request) -> dict[str, Any]:
    store = request.app.state.store
    detector = CommunityDetector(store)
    return {
        "summaries": detector.load_summaries(),
        "stats": detector.community_stats(),
    }


@router.post("/resolve")
def resolve_entities(request: Request) -> dict[str, Any]:
    store = request.app.state.store
    resolver = EntityResolver(store, embed=request.app.state.embed)
    resolver.resolve()
    return resolver.stats()
