from fastapi import APIRouter, Request

from ..models import EvidenceTriple, QueryRequest, QueryResponse
from ...inference.graphrag import GraphRAG

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, request: Request) -> QueryResponse:
    store = request.app.state.store
    rag = GraphRAG(store, model=req.model, hops=req.hops)
    result = rag.ask(req.question)

    cited = [
        EvidenceTriple(
            subject=t["subject"],
            predicate=t["predicate"],
            object=t["object"],
            confidence=t["confidence"],
            source_doc=t.get("source_doc", ""),
            layer=t.get("layer", "source_facts"),
            evidence=t.get("evidence", ""),
        )
        for t in result.get("evidence", [])
    ]

    return QueryResponse(
        question=req.question,
        answer=result["answer"],
        anchor_entities=result.get("anchor_entities", []),
        subgraph_size=result.get("subgraph_size", 0),
        cited_triples=cited,
    )
