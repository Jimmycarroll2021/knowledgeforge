# KnowledgeForge

## What This Is

KnowledgeForge is a domain-agnostic knowledge graph engine. A user clones the repo, drops raw unstructured data into the pipeline, and a validated, queryable, provenance-backed, LLM-assisted knowledge graph is created automatically. Swapping the domain Adapter is the only change needed for a new domain.

## Core Value

Raw unstructured data in → validated, queryable, provenance-backed knowledge graph out. One command. No manual schema.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] REQ-001: Pluggable Adapter interface — new domain = new Adapter only, core unchanged
- [ ] REQ-002: Triple primitive with full provenance (source, relation, target, source_kind, target_kind, evidence, confidence, timestamp)
- [ ] REQ-003: Entity resolution pipeline (blocking → candidates → deterministic rules → embedding similarity → confidence → merge)
- [ ] REQ-004: SHACL shape validation on every triple before graph commit
- [ ] REQ-005: PROV-O provenance on every fact (source record, extraction method, confidence, timestamp, version)
- [ ] REQ-006: Separation of facts vs inference (source facts / normalised triples / resolved entities / inferred relations / LLM explanations — distinct layers)
- [ ] REQ-007: GraphRAG inference — graph-aware retrieval (entities, relations, neighbourhoods, paths, communities) NOT just vector search
- [ ] REQ-008: SQLite graph store (dev), pluggable backend (prod) — no vendor lock-in
- [ ] REQ-009: ChromaDB vector store for node and relation embeddings
- [ ] REQ-010: REST API — ingest, query, graph traversal, provenance lookup
- [ ] REQ-011: Containerised deployment — clone → configure → run, no author required
- [ ] REQ-012: Two working Adapter examples: vault (markdown/Obsidian) and real-estate

### Out of Scope

- LangChain / LlamaIndex wrappers — defeats the "implements what research says" bar
- Single vendor graph database (Neo4j, etc.) — must be pluggable
- Notebook demo — this ships and runs
- SPARQL full spec — SPARQL-style API only (pattern/path queries over SQLite)
- Multi-tenant SaaS — personal/team deployment only, v1

## Context

**Existing fragments to extract (not rebuild):**
- `redact-au/services/redact-au-workspace/src/redact_au_workspace/knowledgegraph.py` — working triple extractor (VaultCandidate, VaultScan, VaultIngestResult, extract_vault_facts, ingest_knowledgegraph_vault)
- `redact-au/services/redact-au-workspace/scripts/ingest_knowledgegraph_vault.py` — working vault ingestion CLI
- `~/.mempalace/palace/chroma.sqlite3` — 37,837 live vector embeddings (ChromaDB)
- `~/.mempalace/knowledge_graph.sqlite3` — 20 entities, 10 triples (working triple store)
- `KnowledgeGraph/` vault — 447 notes, KnowledgeForge 8-file architecture spec

**Research foundation (KG vault, do not recreate):**
- Battaglia et al. (2018) — Relational Inductive Biases, GNNs
- Hamilton et al. (2017) — GraphSAGE, inductive representation learning
- Kipf & Welling (2016) — GCN
- Bordes et al. (2013) — TransE
- Edge et al. (2024) — Microsoft GraphRAG (arXiv:2404.16130)
- W3C RDF 1.1, SHACL, PROV-O

**Public GitHub:** https://github.com/Jimmycarroll2021/jc_knowledgeGrapth (KG vault)
**New repo (to create):** https://github.com/Jimmycarroll2021/knowledgeforge

## Constraints

- **Stack:** Python, uv (not pip), `.venv/Scripts/python.exe` on Windows
- **Graph store:** SQLite (concrete `SQLiteGraphStore`). A pluggable backend is a *goal*, not yet built — extracting a `GraphStore` Protocol is the documented prerequisite for the deferred Neo4j/GDS tier (see Roadmap).
- **Vectors:** ChromaDB (already live in mempalace)
- **LLM:** Anthropic SDK (Claude API) for inference layer
- **Standards:** RDF triple model, SHACL validation, PROV-O provenance — non-negotiable
- **No vendor lock-in (partial):** the **Adapter** layer is behind a Protocol (VaultAdapter/UniversalAdapter). The graph store is *not* yet behind a Protocol — it is the concrete `SQLiteGraphStore`, and several consumers read its `_conn` directly; promoting those to a typed `GraphStore` Protocol is the prerequisite for an alternative backend.
- **Machine:** jimbot — Windows 11 Pro, Git Bash shell

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Extract from redact-au, don't rebuild | Working triple extractor + CLI already exist | — Pending |
| SQLite first, pluggable second | Dev speed; the interface matters more than the backend | SQLite shipped (`SQLiteGraphStore`); the pluggable `GraphStore` Protocol is NOT yet extracted — deferred until the Neo4j tier is justified by scale |
| ChromaDB for vectors | Already live with 37,837 embeddings in mempalace | — Pending |
| Anthropic SDK for LLM | Jim's primary stack; Claude API available | — Pending |
| SHACL over application-layer validation | Standards-aligned; validates graph shape not just data | — Pending |
| Adapter pattern for domain onboarding | Domain swap = Adapter swap only — core frozen | — Pending |

---
*Last updated: 2026-06-21 — project initialized via /gsd-new-project*
