"""Pydantic request/response models — provenance on every response."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TripleOut(BaseModel):
    triple_id: str
    subject: str
    predicate: str
    object: str
    confidence: float
    source_doc: str
    extraction_method: str
    layer: str
    timestamp: str
    evidence: str
    lineage: list[str] = Field(default_factory=list)


class IngestRequest(BaseModel):
    source: str = Field(..., description="Absolute path to source directory or file.")
    adapter: str = Field("universal", description="Adapter: 'vault' or 'universal'.")
    max_facts: int = Field(40, ge=1, le=500)
    dry_run: bool = False


class IngestResponse(BaseModel):
    adapter: str
    source: str
    documents_scanned: int
    documents_extracted: int
    triples_added: int
    triples_skipped: int
    errors: list[str]


class QueryRequest(BaseModel):
    question: str
    hops: int = Field(2, ge=1, le=4)
    model: str = "claude-haiku-4-5-20251001"
    mode: str = Field(
        "local",
        pattern="^(local|global|drift)$",
        description="local = k-hop subgraph retrieval; global = community-summary synthesis (Edge et al. 2024); "
        "drift = community themes + local retrieval fused (Microsoft GraphRAG)",
    )


class EvidenceTriple(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float
    source_doc: str
    layer: str
    evidence: str
    timestamp: str = ""
    lineage: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    question: str
    answer: str
    anchor_entities: list[str]
    subgraph_size: int
    cited_triples: list[EvidenceTriple]
    mode: str = "local"
    communities_used: int = 0


class EmbedRequest(BaseModel):
    batch_size: int = Field(64, ge=1, le=512)
    model: str = "all-MiniLM-L6-v2"


class EmbedResponse(BaseModel):
    entities_embedded: int
    skipped: int
    model: str


class SimilarResult(BaseModel):
    entity_id: str
    kind: str | None
    score: float


class SimilarResponse(BaseModel):
    entity: str
    results: list[SimilarResult]


class NodeResponse(BaseModel):
    entity_id: str
    kind: str
    triples: list[TripleOut]


class ProvenanceResponse(BaseModel):
    entity: str
    triples: list[TripleOut]


class PathEdge(BaseModel):
    subject: str
    predicate: str
    object: str


class PathResponse(BaseModel):
    from_entity: str
    to_entity: str
    path: list[PathEdge]
    found: bool


class GraphStatsResponse(BaseModel):
    entities: int
    triples: int
    by_layer: dict[str, int]
    top_predicates: dict[str, int]


class HealthResponse(BaseModel):
    status: str
    entities: int
    triples: int
    embedded_entities: int
    db_path: str
    embeddings_path: str
