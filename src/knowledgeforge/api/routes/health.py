from fastapi import APIRouter, Request

from ..models import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    store = request.app.state.store
    embed = request.app.state.embed
    stats = store.stats()
    embed_stats = embed.stats()
    return HealthResponse(
        status="ok",
        entities=stats["entities"],
        triples=stats["triples"],
        embedded_entities=embed_stats["embedded_entities"],
        db_path=request.app.state.db_path,
        embeddings_path=request.app.state.embeddings_path,
    )
