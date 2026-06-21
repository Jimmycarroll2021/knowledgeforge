from fastapi import APIRouter, Request

from ..models import EmbedRequest, EmbedResponse, SimilarResponse, SimilarResult
from ...embeddings.pipeline import EmbeddingPipeline

router = APIRouter()


@router.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest, request: Request) -> EmbedResponse:
    pipeline: EmbeddingPipeline = request.app.state.embed
    result = pipeline.embed_all(batch_size=req.batch_size)
    return EmbedResponse(
        entities_embedded=result["entities_embedded"],
        skipped=result["skipped"],
        model=result["model"],
    )


@router.get("/similar/{entity_id:path}", response_model=SimilarResponse)
def similar(
    entity_id: str,
    request: Request,
    k: int = 10,
    kind: str | None = None,
    text: bool = False,
) -> SimilarResponse:
    pipeline: EmbeddingPipeline = request.app.state.embed
    if text:
        raw = pipeline.search_by_text(entity_id, k=k, kind_filter=kind)
    else:
        raw = pipeline.find_similar(entity_id, k=k, kind_filter=kind)
    return SimilarResponse(
        entity=entity_id,
        results=[SimilarResult(**r) for r in raw],
    )
