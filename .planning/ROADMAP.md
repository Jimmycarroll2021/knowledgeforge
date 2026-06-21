# KnowledgeForge — Roadmap

**Goal:** Raw unstructured data in → validated, queryable, provenance-backed knowledge graph out. One command.  
**Success bar:** Best-in-class — implements what the research actually says (RDF, SHACL, PROV-O, real GraphRAG).

---

## Phase 1 — Foundation
**Goal:** Repo scaffold, extract fragments from redact-au, define canonical data contracts.  
**Duration:** ~2 days  
**Covers:** REQ-001, REQ-002

### Tasks
1. Init repo: `pyproject.toml` (uv), src layout, `.gitignore`, `.env.example`
2. Extract `knowledgegraph.py` from redact-au → `src/knowledgeforge/adapter/vault.py`
3. Extract `ingest_knowledgegraph_vault.py` → `src/knowledgeforge/cli/ingest.py`
4. Define canonical `Triple` dataclass with all provenance fields
5. Define `Adapter` Protocol (scan, extract, schema)
6. Define `SourceDocument`, `AdapterSchema`, `RawTriple` contracts
7. Write `AdapterSchema` SHACL shape generator stub
8. Unit tests: Triple serialization, Adapter Protocol compliance
9. GitHub repo `Jimmycarroll2021/knowledgeforge` + initial push

### Gate
- `uv run pytest tests/` green
- `Triple` dataclass with all 11 provenance fields
- VaultAdapter wraps extracted code and passes protocol tests

---

## Phase 2 — Triple Engine
**Goal:** Graph store with SHACL validation, PROV-O provenance, fact/inference layer separation.  
**Duration:** ~3 days  
**Covers:** REQ-004, REQ-005, REQ-006, REQ-008

### Tasks
1. `GraphStore` Protocol: add_triple, query, traverse, validate, provenance_lookup
2. SQLite implementation: `src/knowledgeforge/store/sqlite.py`
   - Schema: entities, triples (with layer column), provenance tables
   - WAL mode, connection pooling
3. InMemory stub implementation (validates Protocol is pluggable)
4. SHACL validator: pyshacl wrapper, shapes from AdapterSchema
5. PROV-O provenance recorder: wasGeneratedBy, wasDerivedFrom, wasAttributedTo
6. Layer separation: source_facts / normalised_triples / inferred_relations / llm_hypotheses
7. Ingestion pipeline: Adapter.scan → Adapter.extract → SHACL validate → store
8. CLI: `knowledgeforge ingest --adapter vault --source <path> [--dry-run]`
9. Integration tests: vault adapter → SQLite store, end-to-end

### Gate
- Invalid triple rejected with SHACL violation report
- Valid vault ingestion: triples in SQLite with full provenance
- Provenance query: any triple → source_doc + extraction_method + timestamp
- Two GraphStore implementations passing same contract tests

---

## Phase 3 — Entity Resolution
**Goal:** Blocking → candidate generation → rules → embedding similarity → confidence → merge.  
**Duration:** ~3 days  
**Covers:** REQ-003

### Tasks
1. Blocking: name-prefix + type-based blocking keys
2. Candidate generation: pairs within blocks
3. Deterministic rules: exact match, alias normalization, canonical forms
4. Sentence-transformer embeddings for entity descriptions (all-MiniLM-L6-v2)
5. Cosine similarity scoring on embedding pairs
6. Confidence weighting: rule_score * 0.6 + embedding_score * 0.4
7. Merge policy: merge ≥ 0.85, flag 0.70–0.85, reject < 0.70
8. Entity registry: canonical_id → aliases, merged_from, confidence
9. Post-resolution: normalised_triples layer populated from source_facts
10. Test harness: held-out vault entity pairs with known duplicates → F1 score

### Gate
- Entity resolution F1 ≥ 0.85 on held-out test set
- Merged entities traceable to source entities via provenance

---

## Phase 4 — Embedding Pipeline
**Goal:** Node + relation embeddings in ChromaDB. GraphSAGE-style structural aggregation.  
**Duration:** ~2 days  
**Covers:** REQ-009

### Tasks
1. ChromaDB client: `src/knowledgeforge/embeddings/store.py`
   - Collections: `entities`, `relations`, `communities`
2. Semantic embeddings: sentence-transformer on entity_description + top evidence
3. GraphSAGE aggregation (simplified): mean-pool 1-hop neighbour embeddings
4. Relation embeddings: TransE-style subject + predicate → object vector
5. Batch embedding job: `knowledgeforge embed --store <path>`
6. Similarity search: `find_similar_entities(entity_id, k=10)`
7. Integration: entity resolution uses embedding similarity from ChromaDB

### Gate
- `find_similar_entities("GraphSAGE", k=5)` returns semantically related graph ML entities
- All entities in vault graph have embeddings in ChromaDB

---

## Phase 5 — GraphRAG Inference Engine
**Goal:** Graph-aware retrieval + LLM answer grounding. NOT vector search.  
**Duration:** ~3 days  
**Covers:** REQ-007

### Tasks
1. Entity extraction from query (regex + ChromaDB lookup)
2. Neighbourhood expansion: BFS 1-2 hops from extracted entities
3. Path finding: shortest path between entity pairs (Dijkstra on SQLite graph)
4. Community detection: simplified Leiden (or greedy modularity) on SQLite graph
5. Community summaries: LLM-generated (Claude API, cached per community)
6. Context assembly: entities + relations + paths + community summaries → prompt context
7. LLM grounded answer: Claude API with graph context, returns answer + cited_triples
8. `knowledgeforge query "<question>"` CLI command
9. Tests: query returns cited_triples from graph (not embedding chunks)

### Gate
- Query "what connects GraphSAGE to entity resolution?" returns graph path as evidence
- Answer cites source triples with provenance (not raw text chunks)
- Community summaries cached in ChromaDB, not re-generated on each query

---

## Phase 6 — Query API
**Goal:** FastAPI REST API. Containerised. Clone → configure → run.  
**Duration:** ~2 days  
**Covers:** REQ-010, REQ-011

### Tasks
1. FastAPI app: `src/knowledgeforge/api/main.py`
   - POST /ingest, POST /query, GET /graph/node/{id}, GET /graph/path, GET /provenance/{id}, GET /health
2. Request/response models: Pydantic, all responses include provenance fields
3. Adapter registry: config.yaml maps adapter names to classes
4. Dockerfile: Python 3.12, uv, non-root user
5. docker-compose.yml: app + ChromaDB (named volume)
6. config.example.yaml + .env.example
7. README: clone → configure → run in under 10 commands
8. API integration tests (httpx test client)

### Gate
- `docker-compose up` produces running API at localhost:8000
- POST /query returns `cited_triples` array with provenance
- GET /provenance/{id} returns full PROV-O lineage

---

## Phase 7 — Evaluation & Docs
**Goal:** Adapter examples, research alignment, evaluation results, production checklist.  
**Duration:** ~2 days  
**Covers:** REQ-012

### Tasks
1. VaultAdapter: clean extraction from redact-au, full docstring + schema
2. RealEstateAdapter: maps PropertyPeek listing fields to triples
3. Evaluation report: entity resolution F1, retrieval quality (MRR@10), latency benchmarks
4. Research alignment doc: maps each layer to primary literature (Battaglia, Hamilton, Kipf, Bordes, Edge)
5. Architecture doc: design choices, data contracts, tradeoffs
6. Production checklist: observability (structured logging), secrets (env vars), CI (GitHub Actions), data governance
7. GitHub Actions: lint (ruff), typecheck (mypy), test (pytest) on push

### Gate
- Both adapters run end-to-end on sample data
- Research alignment doc maps each layer to a primary source
- CI passes on clean clone

---

## Milestone Summary

| Phase | Focus | Duration | Key Gate |
|-------|-------|----------|----------|
| 1 | Foundation — contracts + fragment extraction | 2d | Triple + Adapter Protocol tests green |
| 2 | Triple Engine — store + SHACL + PROV-O | 3d | End-to-end vault ingest with full provenance |
| 3 | Entity Resolution — blocking → merge | 3d | F1 ≥ 0.85 on held-out set |
| 4 | Embedding Pipeline — ChromaDB + GraphSAGE | 2d | Semantic similarity working on vault entities |
| 5 | GraphRAG — graph-aware retrieval + LLM | 3d | Answers cite graph triples (not chunks) |
| 6 | Query API — FastAPI + Docker | 2d | clone → docker-compose up → running API |
| 7 | Evaluation + Docs — research alignment | 2d | CI green, two working Adapters |

**Total estimated: ~17 working days**

---

## Research Alignment (non-negotiable)

Every layer maps to primary literature already in the KG vault:

| Layer | Paper | Key concept |
|-------|-------|-------------|
| Triple Primitive | W3C RDF 1.1 | S→P→O graph model |
| SHACL Validation | W3C SHACL | Shape constraint language |
| PROV-O Provenance | W3C PROV-O | wasGeneratedBy, wasDerivedFrom |
| Entity Resolution | Vault concept notes | Blocking, embedding similarity, merge policy |
| Embedding Pipeline | Hamilton 2017 (GraphSAGE) | Inductive neighbourhood aggregation |
| Relation Embeddings | Bordes 2013 (TransE) | Translation-based KG embeddings |
| GraphRAG | Edge 2024 (arXiv:2404.16130) | Community-aware graph retrieval |
| GNN Foundation | Battaglia 2018 | Relational inductive biases |
