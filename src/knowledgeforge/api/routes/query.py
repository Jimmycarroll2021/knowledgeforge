from fastapi import APIRouter, Request

from ..models import EvidenceTriple, QueryRequest, QueryResponse
from ...inference.graphrag import GraphRAG
from ...resolution.resolver import EntityResolver

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, request: Request) -> QueryResponse:
    store = request.app.state.store
    embed = request.app.state.embed
    rag = GraphRAG(
        store,
        model=req.model,
        hops=req.hops,
        embed=embed,
        resolver=EntityResolver(store, embed=embed),
    )
    result = rag.ask(req.question, mode=req.mode)

    cited = [
        EvidenceTriple(
            subject=t["subject"],
            predicate=t["predicate"],
            object=t["object"],
            confidence=t["confidence"],
            source_doc=t.get("source_doc", ""),
            layer=t.get("layer", "source_facts"),
            evidence=t.get("evidence", ""),
            timestamp=t.get("timestamp", ""),
            lineage=t.get("lineage", []),
        )
        for t in result.get("evidence", [])
    ]

    return QueryResponse(
        question=req.question,
        answer=result["answer"],
        anchor_entities=result.get("anchor_entities", []),
        subgraph_size=result.get("subgraph_size", 0),
        cited_triples=cited,
        mode=result.get("mode", req.mode),
        communities_used=result.get("communities_used", 0),
    )
