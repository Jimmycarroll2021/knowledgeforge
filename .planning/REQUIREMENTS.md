# KnowledgeForge — Requirements

## REQ-001: Pluggable Adapter Interface
**Priority:** P0 — Core architecture constraint  
**Phase:** 1 (Foundation)

A domain `Adapter` is the ONLY thing a user writes to onboard a new data source. The core engine is frozen. The Adapter maps domain-specific schemas to the canonical `Triple` contract.

**Contract:**
```python
class Adapter(Protocol):
    def scan(self, source_path: Path) -> list[SourceDocument]: ...
    def extract(self, doc: SourceDocument) -> list[RawTriple]: ...
    def schema(self) -> AdapterSchema: ...  # declared predicates, entity types
```

**Acceptance:** New domain onboarded by writing one Adapter class. Zero changes to engine code.

---

## REQ-002: Triple Primitive with Full Provenance
**Priority:** P0  
**Phase:** 1 (Foundation)

Every fact in the system is a `Triple`:

```python
@dataclass
class Triple:
    subject: str          # entity URI or local ID
    predicate: str        # relation type (from Adapter schema)
    object: str           # entity URI, local ID, or literal
    source_kind: str      # entity type of subject
    target_kind: str      # entity type of object
    evidence: str         # verbatim text supporting this triple
    confidence: float     # 0.0–1.0
    source_doc: str       # originating document path/URI
    extraction_method: str # "rule" | "llm" | "structural"
    timestamp: str        # ISO-8601
    version: str          # extractor version
```

**Acceptance:** No triple enters the graph store without all provenance fields. Facts and LLM inferences stored in separate layers.

---

## REQ-003: Entity Resolution Pipeline
**Priority:** P0  
**Phase:** 3 (Entity Resolution)

Pipeline stages:
1. **Blocking** — reduce candidate pairs via blocking keys (name prefix, type)
2. **Candidate generation** — pairs within each block
3. **Deterministic rules** — exact match, alias lookup, canonical normalization
4. **Embedding similarity** — sentence-transformer cosine similarity on entity descriptions
5. **Confidence scoring** — weighted combination of rule + embedding signals
6. **Merge policy** — merge above threshold, flag for review between bounds, reject below

**Acceptance:** Entity resolution F1 ≥ 0.85 on a held-out test set (vault entities vs. known duplicates).

---

## REQ-004: SHACL Shape Validation
**Priority:** P0  
**Phase:** 2 (Triple Engine)

Every triple batch is validated against SHACL shapes before graph commit. Shapes are declared by the Adapter via its `schema()` method. Invalid triples are rejected with structured error reports (not silently dropped).

**Acceptance:** Invalid triple (missing predicate, wrong type) is rejected with SHACL violation report. Valid triple commits cleanly.

---

## REQ-005: PROV-O Provenance
**Priority:** P0  
**Phase:** 2 (Triple Engine)

Every graph entity traces to a PROV-O `Entity` with `wasGeneratedBy` (extraction activity), `wasDerivedFrom` (source document), and `wasAttributedTo` (extractor version). API provenance endpoint returns full lineage for any node or triple.

**Acceptance:** Any graph node can be traced to its source document, extraction method, and timestamp via the provenance API.

---

## REQ-006: Fact / Inference Separation
**Priority:** P0  
**Phase:** 2 (Triple Engine)

Four distinct layers in the graph store:
- `source_facts` — raw triples from Adapter extraction
- `normalised_triples` — post-entity-resolution, canonical form
- `inferred_relations` — derived by graph traversal rules
- `llm_hypotheses` — LLM-generated, clearly flagged, never mixed with facts

**Acceptance:** API query can filter by layer. LLM hypotheses never appear in fact-layer results.

---

## REQ-007: GraphRAG Inference Engine
**Priority:** P0  
**Phase:** 5 (GraphRAG)

Graph-aware retrieval pipeline:
1. Entity extraction from query
2. Node lookup + neighbourhood expansion (1-2 hops)
3. Path finding between entities
4. Community detection (Leiden algorithm)
5. Context assembly (entities + relations + paths + community summaries)
6. LLM answer grounded in graph context

NOT a vector similarity search over flattened text. The graph structure is the retrieval mechanism.

**Acceptance:** Query "what connects X to Y?" returns path evidence from graph. Query answers cite source triples, not embedding chunks.

---

## REQ-008: Pluggable Graph Store
**Priority:** P1  
**Phase:** 2 (Triple Engine)

Graph store behind a `GraphStore(Protocol)` interface. SQLite implementation ships. Interface supports: add_triple, query (subject/predicate/object pattern), traverse (BFS/DFS), validate (SHACL), provenance_lookup.

**Acceptance:** SQLite implementation passes full interface contract tests. A stub second implementation (in-memory) validates the interface is genuinely pluggable.

---

## REQ-009: ChromaDB Embedding Pipeline
**Priority:** P1  
**Phase:** 4 (Embedding Pipeline)

Node and relation embeddings:
- Semantic: sentence-transformer on entity description + evidence
- Structural: GraphSAGE-style aggregation from neighbours (simplified)
- Stored in ChromaDB collections: `entities`, `relations`, `communities`

**Acceptance:** Entity similarity search returns semantically related entities. GraphSAGE neighbourhood aggregation runs on the SQLite graph.

---

## REQ-010: REST Query API
**Priority:** P1  
**Phase:** 6 (Query API)

FastAPI endpoints:
- `POST /ingest` — run Adapter pipeline on source path
- `POST /query` — GraphRAG query, returns answer + cited triples
- `GET /graph/node/{id}` — entity + neighbours
- `GET /graph/path` — shortest path between two entities
- `GET /provenance/{id}` — full PROV-O lineage for node or triple
- `GET /health` — system status

**Acceptance:** All endpoints return JSON with provenance fields. `/query` response includes `cited_triples` array.

---

## REQ-011: Containerised Deployment
**Priority:** P1  
**Phase:** 6 (Query API)

`Dockerfile` + `docker-compose.yml`. `clone → cp config.example.yaml config.yaml → docker-compose up` = running system. No author intervention required. `.env.example` for all secrets.

**Acceptance:** Clean clone + docker-compose up produces a running API at localhost:8000. Works on a machine with no prior context.

---

## REQ-012: Working Adapter Examples
**Priority:** P1  
**Phase:** 7 (Evaluation & Docs)

Two adapter examples:
1. **VaultAdapter** — Obsidian/markdown vault (extracted from redact-au knowledgegraph.py)
2. **RealEstateAdapter** — property listing data (from PropertyPeek structure)

**Acceptance:** Both adapters run end-to-end on sample data, producing validated triples in the graph store.

---

## Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-001 | Triple extraction throughput | ≥ 1,000 triples/min on local hardware |
| NFR-002 | Query latency (GraphRAG) | < 5s p95 on graphs up to 100K nodes |
| NFR-003 | Entity resolution F1 | ≥ 0.85 on held-out set |
| NFR-004 | Graph validation | SHACL rejection rate 0% on valid triples |
| NFR-005 | Provenance coverage | 100% of triples have full lineage |
