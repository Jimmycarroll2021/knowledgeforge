# KnowledgeForge — Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-21)

**Core value:** Raw unstructured data in → validated, queryable, provenance-backed knowledge graph out. One command.  
**Current focus:** Phase 4 — Embedding Pipeline

---

## Phase Status

| Phase | Name | Status | Notes |
|-------|------|--------|-------|
| 1 | Foundation | ✅ COMPLETE | Triple + Adapter Protocol, 35 tests green, VaultAdapter + UniversalAdapter |
| 2 | Triple Engine | ✅ COMPLETE (partial) | SQLite store, provenance, pipeline, CLI, ingest. Missing: SHACL pyshacl validation |
| 3 | Entity Resolution | ✅ COMPLETE | 3-phase (exact → Jaro-Winkler → WCC), SAME_AS edges, entity_aliases table |
| 4 | Embedding Pipeline | ❌ NOT STARTED | ChromaDB embeddings not implemented |
| 5 | GraphRAG | ✅ COMPLETE | BFS subgraph, LLM grounding, path finding, OAuth CLI fallback, tested |
| 6 | Query API | ❌ NOT STARTED | FastAPI + Docker not implemented |
| 7 | Evaluation & Docs | ❌ NOT STARTED | Benchmarks not run |

**Bonus (not in roadmap):**
- LLM semantic triple extractor (claude-p CLI backend) — COMPLETE
  - Uses Claude to extract typed (subject, predicate, object) triples per section
  - Extractions running on algorithms/, concepts/, paper-notes/ (in progress 2026-06-21)
  - 39 LLM triples added so far; extraction ongoing

---

## Live Graph State (2026-06-21)

- **Documents ingested:** 83 (rule-based) + ongoing LLM extraction
- **Entities:** 961
- **Triples:** 1720
  - Rule-based (structural): 1681 (CONTAINS_HEADING, LINKS_TO, etc.)
  - LLM semantic: 39 (PROPOSED_BY, DEFINED_AS, IMPROVES_ON, etc. — growing)
- **Graph store:** `knowledgeforge/data/graph.db` (SQLite WAL)
- **Extraction jobs running:** algorithms/ (6 files), concepts/ (22 files), paper-notes/ (9 files)

---

## Key Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-06-21 | Extract from redact-au, don't rebuild | Working triple extractor + CLI already exist |
| 2026-06-21 | SQLite first, pluggable interface | Protocol/ABC abstraction is the deliverable, not the backend |
| 2026-06-21 | Skip research phase | Deep domain knowledge exists in KG vault (447 notes) |
| 2026-06-21 | No LangChain/LlamaIndex | Defeats "implements what research says" bar |
| 2026-06-21 | Add LLM extractor phase | Rule-based CONTAINS_HEADING/LINKS_TO is structural, not semantic. 90.4% noise before noise dir exclusion |
| 2026-06-21 | OAuth CLI fallback for LLM calls | Jim uses Claude Code OAuth (no separate API key). `claude -p` works with OAuth auth |
| 2026-06-21 | Noise dirs excluded from UniversalAdapter | tools/, scripts/, migration/, system/ contain operational data not knowledge |

---

## Blockers

None.

---

## Next Actions

1. Wait for LLM extraction jobs (algorithms/concepts/paper-notes) to complete
2. Verify semantic triples — re-run `query "how does TransE work?"` to see improvement
3. Run entity resolution: `knowledgeforge resolve` on enriched graph
4. Phase 4: ChromaDB embedding pipeline (GraphSAGE-style node embeddings)
5. Phase 6: FastAPI REST + Docker containerisation
6. Phase 7: Evaluation benchmarks, GitHub Actions CI

---

## Initialized

2026-06-21 via /gsd-new-project  
Last updated: 2026-06-21 (Phase 1-3 + 5 complete, LLM extractor added)
