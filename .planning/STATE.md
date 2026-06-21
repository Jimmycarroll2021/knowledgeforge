# KnowledgeForge — Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-21)

**Core value:** Raw unstructured data in → validated, queryable, provenance-backed knowledge graph out. One command.  
**Current focus:** Phase 7 — Evaluation & Docs

---

## Phase Status

| Phase | Name | Status | Notes |
|-------|------|--------|-------|
| 1 | Foundation | ✅ COMPLETE | Triple + Adapter Protocol, 35 tests green, VaultAdapter + UniversalAdapter |
| 2 | Triple Engine | ✅ COMPLETE | SQLite WAL store, provenance, pipeline, CLI, ingest |
| 3 | Entity Resolution | ✅ COMPLETE | 3-phase (exact → Jaro-Winkler → WCC), 447 aliases written |
| 4 | Embedding Pipeline | ✅ COMPLETE | sentence-transformers + GraphSAGE MEAN agg + turbovec SIMD + ChromaDB |
| 5 | GraphRAG | ✅ COMPLETE | BFS k-hop, grounded LLM answers, path finder, OAuth CLI fallback |
| 6 | Query API | ✅ COMPLETE | FastAPI + Docker, `knowledgeforge serve`, gate verified |
| 7 | Evaluation & Docs | ❌ NOT STARTED | Benchmarks, CI, research alignment doc — NEXT |

---

## Live Graph State (2026-06-21)

- **Documents ingested:** 83 (rule-based) + LLM extraction across all dirs
- **Entities:** 1,125
- **Triples:** 1,891
  - Structural (CONTAINS_HEADING, LINKS_TO, etc.): 1,681
  - LLM semantic (PROPOSED_BY, DEFINED_AS, TYPE_OF, etc.): 210
- **Entity aliases:** 447 (entity resolution complete)
- **Embeddings:** 1,059 entities in ChromaDB + turbovec index built
- **Graph store:** `knowledgeforge/data/graph.db` (SQLite WAL)
- **Embeddings store:** `knowledgeforge/data/embeddings/` (ChromaDB)

---

## CLI Commands (all working)

```bash
knowledgeforge ingest --source <path>          # rule-based structural extraction
knowledgeforge extract --source <path>         # LLM semantic triple extraction
knowledgeforge resolve                          # 3-phase entity resolution
knowledgeforge embed                            # GraphSAGE + turbovec embeddings
knowledgeforge similar <entity>                 # fast SIMD similarity search
knowledgeforge query "<question>"              # GraphRAG grounded answer
knowledgeforge query x --path-only --from-entity A --to-entity B  # path find
knowledgeforge stats                            # graph statistics
knowledgeforge provenance <entity>             # full provenance for entity
```

---

## Key Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-06-21 | Extract from redact-au, don't rebuild | Working triple extractor + CLI already exist |
| 2026-06-21 | SQLite first, pluggable interface | Protocol/ABC abstraction is the deliverable |
| 2026-06-21 | Skip research phase | Deep domain knowledge in KG vault (447 notes) |
| 2026-06-21 | No LangChain/LlamaIndex | Defeats "implements what research says" bar |
| 2026-06-21 | LLM extractor added | Rule-based CONTAINS_HEADING/LINKS_TO is structural not semantic |
| 2026-06-21 | OAuth CLI fallback for LLM calls | Jim uses Claude Code OAuth, no separate API key needed |
| 2026-06-21 | turbovec for embedding similarity | Google TurboQuant 4-bit SIMD — not a competitor, complementary to ChromaDB |
| 2026-06-21 | GraphSAGE MEAN aggregation | Hamilton 2017 h_v = MEAN(h_v ∪ {h_u ∀u ∈ N(v)}) |

---

## Phase 6 — Next Actions

FastAPI REST API + Docker:

```
src/knowledgeforge/api/
  main.py       — FastAPI app, lifespan, CORS
  routes/
    ingest.py   — POST /ingest
    query.py    — POST /query (GraphRAG)
    graph.py    — GET /graph/node/{id}, GET /graph/path, GET /provenance/{id}
    embed.py    — POST /embed, GET /similar/{entity}
  models.py     — Pydantic request/response with provenance fields
Dockerfile      — Python 3.12, uv, non-root user
docker-compose.yml — app + ChromaDB named volume
```

Gate: `docker-compose up` → running API at localhost:8000  
POST /query returns `cited_triples` array with provenance.

---

## Initialized

2026-06-21 via /gsd-new-project  
Last updated: 2026-06-21 (Phases 1-5 complete, 6-7 next)
