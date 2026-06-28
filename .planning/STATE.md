# KnowledgeForge — Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-21)

**Core value:** Raw unstructured data in → validated, queryable, provenance-backed knowledge graph out. One command.
**Current focus:** Gap-closure build COMPLETE — all 11 P0+P1 audit gaps closed; full test suite green.

---

## Gap-Closure Build (2026-06-21)

A quality build closed 11 P0+P1 audit gaps whose purpose was to make previously-aspirational
claims TRUE in the code. Every item below is verified against source, not asserted.

| # | Gap | Resolution | Where |
|---|-----|------------|-------|
| 1 | GraphSAGE was a parameter-free average | **Learned** aggregator: trains weight matrix `W` via unsupervised GraphSAGE graph loss (random-walk positives + negative sampling, hand-derived backprop); `z = L2(ReLU(W·CONCAT(self, mean_nbrs)))` | `embeddings/pipeline.py` `_train_aggregator` / `_apply_aggregator` |
| 2 | Entity resolution marked complete but UNMEASURED | **Measured F1 = 1.00** on the 49-pair benchmark (held-out generalisation set not yet in CI); metaphone blocking, initialism rule, jaro+cosine with 0.90 semantic floor, flag band, transitive Union-Find, SAME_AS inferred edges | `resolution/resolver.py`, `tests/test_resolution_eval.py`, `tests/fixtures/er_labelled_pairs.json` |
| 3 | Provenance fields not full PROV-O | `source` (producing agent), `lineage`, `valid_from`/`valid_to`, `schema_version` on every triple; resolver SAME_AS edges carry `source=EntityResolver` | `contracts.py`, `resolution/resolver.py` |
| 4 | "Validation" was predicate-set membership only | SHACL-style `PropertyShape` (cardinality/target-class/datatype/severity); severity ladder with opt-in `strict` mode (default soft) | `contracts.py`, `pipeline.py` `_validate` |
| 5 | Layers not real named graphs | `layer` is part of triple identity + CHECK constraint; `purge_layer` + recursive-CTE `neighbourhood`; vault separated 1068 source_facts + 73 inferred_relations | `store/sqlite.py` |
| 6 | GraphRAG local only | Local k-hop with semantic anchors + trusted-layer filter (excludes `llm_hypotheses`); **global** mode via hierarchical-Leiden communities + LLM summaries (rank top-K → single synthesis, not yet map-reduce); **drift**-style mode (themes + local retrieval fusion); reachable over HTTP via `mode` field | `inference/graphrag.py`, `community/detector.py` |
| 7 | API surface incomplete | `/health /ingest /query /embed /similar /graph/stats /graph/node /graph/path /graph/provenance /graph/community /graph/resolve` | `api/routes/*` |
| 8 | No security/observability | Config-gated API-key auth (`KF_API_KEY`), rate limiting (`KF_RATE_LIMIT`), structured JSON request logging + `X-Request-ID` | `api/security.py` |

Deferred (NOT yet built — single-machine scope): Neo4j/GDS backend tier, **level-aware global
search** over the community hierarchy (+ true map-reduce synthesis, the full DRIFT primer→follow-up→refine loop, and an answer-quality eval harness: retrieval MRR/nDCG + faithfulness), TransE relation-as-first-class-vector
scorer, inductive new-node embedding path, pluggable adapter registry, and a pluggable `GraphStore`
Protocol (the prerequisite for the Neo4j tier — the store is currently the concrete
`SQLiteGraphStore`). Per-build N×N GraphSAGE matrix and O(n²) blocking are fine for
hundreds–low-thousands of entities, not millions.

**v2 (2026-06-27) — Hierarchical GraphRAG spike:** community detection upgraded from flat Louvain
to **hierarchical Leiden** (leidenalg, seeded/deterministic; `level` + `parent_community_id` on every
community, coarse level 0 → finer children, Louvain fallback retained) and a new **DRIFT** query mode
(`query --mode drift`) fuses community themes with local entity retrieval. 61 tests green; mypy --strict
clean. Remaining: make global search level-aware and measure it (MRR@10).

---

## Phase Status

| Phase | Name | Status | Notes |
|-------|------|--------|-------|
| 1 | Foundation | ✅ COMPLETE | Triple + Adapter Protocol, VaultAdapter + UniversalAdapter |
| 2 | Triple Engine | ✅ COMPLETE | SQLite WAL store, PROV-O provenance, named layers, pipeline, CLI |
| 3 | Entity Resolution | ✅ SHIPPED + **MEASURED F1 = 1.00** | Was wrongly marked "complete" while unmeasured. Now: metaphone blocking → jaro+cosine (0.90 semantic floor) → Union-Find → SAME_AS; benchmark F1 = 1.00 (gate ≥ 0.85); held-out generalisation set not yet in CI |
| 4 | Embedding Pipeline | ✅ COMPLETE | **Learned** GraphSAGE aggregator (unsupervised graph loss, hand-derived backprop) + turbovec SIMD + ChromaDB |
| 5 | GraphRAG | ✅ COMPLETE | Local k-hop (trusted-layer grounding) + global community synthesis; path finder; OAuth CLI fallback |
| 6 | Query API | ✅ COMPLETE | FastAPI + Docker; resolve/community endpoints; config-gated auth/rate-limit/logging |
| 7 | Evaluation & Docs | ✅ COMPLETE | Resolution-eval + security tests; research-alignment + production-checklist docs; README rewritten to verified reality |

---

## Live Graph State (2026-06-21 — gap-closure proof run, 73-doc research vault)

- **Documents ingested:** 73 (graph-ML research vault)
- **Entities:** 511
- **Triples:** 1,141
  - Source facts: 1,068
  - Inferred relations (incl. SAME_AS): 73
- **Entity aliases (SAME_AS):** 73 — measured ER F1 = 1.00 on benchmark
- **Embeddings:** all 511 entities embedded via learned GraphSAGE in ~25s; turbovec index built
- **Communities:** 16 detected (Louvain) + LLM-summarised
- **Local query:** grounded, cited answer; correctly refused an unsupported comparison
- **Global query:** synthesised the corpus's three method families
- **Graph store:** `knowledgeforge/data/graph.db` (SQLite WAL)
- **Embeddings store:** `knowledgeforge/data/embeddings/` (ChromaDB) + `graphsage_w.npy` (cached learned W)

---

## CLI Commands (all working)

```bash
knowledgeforge ingest --source <path>          # rule-based structural extraction
knowledgeforge extract --source <path>         # LLM semantic triple extraction
knowledgeforge resolve                          # metaphone → jaro+cosine → Union-Find → SAME_AS
knowledgeforge embed                            # learned GraphSAGE + turbovec embeddings
knowledgeforge community                        # hierarchical Leiden communities + LLM summaries
knowledgeforge similar <entity>                 # fast SIMD similarity search
knowledgeforge query "<question>"              # GraphRAG local grounded answer
knowledgeforge query "<question>" --mode global # community-summary synthesis (Edge et al. 2024)
knowledgeforge query "<question>" --mode drift  # DRIFT: community themes + local retrieval fused
knowledgeforge query x --path-only --from-entity A --to-entity B  # path find (no LLM)
knowledgeforge stats                            # graph statistics
knowledgeforge provenance <entity>             # full provenance for entity
knowledgeforge serve                            # REST API on localhost:8000
```

---

## Key Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-06-21 | Extract from redact-au, don't rebuild | Working triple extractor + CLI already existed |
| 2026-06-21 | SQLite first, pluggable interface | SQLite shipped as concrete `SQLiteGraphStore`; the pluggable `GraphStore` Protocol/ABC is NOT yet extracted (deferred prerequisite for the Neo4j tier) — the **Adapter** layer is the Protocol that shipped |
| 2026-06-21 | No LangChain/LlamaIndex | Defeats "implements what research says" bar |
| 2026-06-21 | LLM extractor added | Rule-based CONTAINS_HEADING/LINKS_TO is structural not semantic |
| 2026-06-21 | OAuth CLI fallback for LLM calls | Jim uses Claude Code OAuth, no separate API key needed |
| 2026-06-21 | turbovec for embedding similarity | Google TurboQuant 4-bit SIMD — complementary to ChromaDB |
| 2026-06-21 | GraphSAGE made a **learned** aggregator | Parameter-free average did not implement Hamilton 2017; now trains `W` via the unsupervised graph loss |
| 2026-06-21 | Entity resolution **measured**, not asserted | "Complete but unmeasured" is an overclaim; added labelled benchmark + F1 gate ≥ 0.85 |
| 2026-06-21 | GraphRAG global mode added | Edge et al. 2024 — local k-hop alone cannot answer corpus-wide thematic questions |
| 2026-06-21 | Security config-gated + OFF by default | Keeps dev/tests unchanged while making prod hardening real |

---

## Initialized

2026-06-21 via /gsd-new-project
Last updated: 2026-06-21 (gap-closure build complete — all 11 P0+P1 gaps closed, docs verified to code)
