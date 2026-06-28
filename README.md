# KnowledgeForge

> Raw unstructured data in → validated, queryable, provenance-backed knowledge graph out.

One command. No manual schema. No vendor lock-in. Domain swap = adapter swap only.

[![CI](https://github.com/Jimmycarroll2021/knowledgeforge/actions/workflows/ci.yml/badge.svg)](https://github.com/Jimmycarroll2021/knowledgeforge/actions/workflows/ci.yml)

---

## What it does

Drop any directory of files (Markdown, PDF, DOCX, HTML, CSV, JSON, code) into the pipeline. KnowledgeForge extracts structured knowledge as typed triples with full provenance, resolves duplicate entities, learns graph-aware embeddings, detects thematic communities, and answers questions grounded in cited evidence — refusing to answer when the graph has no support.

```bash
knowledgeforge ingest  --source ~/my-notes     # structural triples (fast, no LLM)
knowledgeforge extract --source ~/my-notes     # LLM semantic triples
knowledgeforge resolve                          # entity resolution (measured F1, see below)
knowledgeforge embed                            # learned GraphSAGE embeddings
knowledgeforge community                        # hierarchical Leiden communities + LLM summaries
knowledgeforge query "what connects GraphSAGE to entity resolution?"
knowledgeforge query "what are the major method families?" --mode global
knowledgeforge query "how does GraphSAGE relate to entity resolution?" --mode drift
```

Every answer carries `cited_triples` with full provenance: source document, layer, confidence, extraction method, and timestamp.

---

## Scope (read this first)

KnowledgeForge is **single-machine, hundreds-to-low-thousands of entities**. The whole stack — a per-build N×N GraphSAGE training matrix, O(n²) candidate blocking with phonetic/prefix keys, SQLite WAL storage, in-process hierarchical Leiden — is sized for research vaults and personal/team knowledge bases, not for millions of nodes. The path to that scale is real but **not yet built**; see [Roadmap](#roadmap-not-yet-built). Nothing in this README claims a capability that is not in the code.

---

## Architecture

```
Raw files (any format)
    ↓
[Adapter]            ← domain swap = adapter swap only; core engine unchanged
    ↓
[Triple Primitive]   ← (subject, predicate, object) + PROV-O provenance + bitemporal validity
    ↓
[SHACL Validation]   ← PropertyShape: cardinality / target-class / datatype / severity ladder
    ↓
[Entity Resolution]  ← metaphone blocking → jaro+cosine (0.90 semantic floor) → Union-Find → SAME_AS
    ↓
[Embedding Pipeline] ← learned GraphSAGE aggregator (unsupervised graph loss) + turbovec SIMD
    ↓
[Graph Store]        ← SQLite WAL, named layers: source_facts / normalised / inferred / llm_hypotheses
    ↓
[Communities]        ← hierarchical Leiden detection (Louvain fallback) + LLM community summaries (cached)
    ↓
[GraphRAG]           ← local k-hop (trusted-layer grounding) + global community synthesis
    ↓
[REST API]           ← FastAPI, provenance on every response, config-gated auth/rate-limit/logging
```

Every design decision maps to primary literature — see [`docs/research-alignment.md`](docs/research-alignment.md).

---

## Proven on the research vault

Run end-to-end against a 73-document graph-ML research vault (this is the actual gap-closure build result, not an illustration):

| Stage | Result |
|-------|--------|
| Ingest + extract | **1,141 triples / 511 entities** (source_facts + inferred layers, cleanly separated) |
| Resolve | **73 `SAME_AS` aliases** written to the `inferred_relations` layer (originals preserved) |
| Embed | learned-GraphSAGE embedding of all **511 entities in ~25s** (single machine, numpy) |
| Community | **16 communities** detected (Louvain at the time; current default is hierarchical Leiden) and LLM-summarised |
| Query (local) | grounded, cited answer; **correctly refused** to hallucinate an unsupported comparison |
| Query (global) | synthesised the corpus's **three method families** from community summaries |

Entity-resolution quality is **measured**, not asserted — see the next section.

---

## Entity resolution — measured, not claimed

Earlier versions marked entity resolution "complete" with no measurement. That is corrected. Resolution is now evaluated against a hand-labelled benchmark of graph-ML entity ids (`tests/fixtures/er_labelled_pairs.json` — 25 true surface variants + 25 confusable negatives such as GraphSAGE/GraphSAINT and TransE/TransR):

- **F1 = 1.00 on the benchmark** (`tests/test_resolution_eval.py`, gate ≥ 0.85, red by design if it regresses)
- **Generalisation:** a held-out generalisation set is **not yet in CI** — only the 50-pair benchmark above is measured and gated. (An informal probe on novel terms looked strong, but it is not a CI-backed number.)

How it works (`src/knowledgeforge/resolution/resolver.py`):

1. **Phase 1 — exact normalised match** within a kind (case / punctuation / spacing only).
2. **Phase 2 — blocked similarity**: metaphone + prefix blocking generates candidates; an initialism rule (e.g. `GNN` ⇄ `Graph Neural Network`) auto-merges deterministically; otherwise a pair must clear a combined `jaro·0.6 + cosine·0.4` score **and** a **0.90 cosine semantic floor**. Borderline pairs land in a *flag band* (recorded, never silently merged). (The `jaro·0.6 + cosine·0.4` cosine stage needs embeddings — run `knowledgeforge embed` first, or use the API/eval path which inject them; the bare `resolve` CLI without embeddings is string-only.)
3. **Phase 3 — structural WCC** over `SIMILAR_TO` edges.

All merges feed one path-compressed **Union-Find** so transitive duplicates collapse to a single canonical root. Every merge writes a `SAME_AS` edge to the `inferred_relations` layer carrying `source=EntityResolver` provenance. Originals are never deleted.

---

## Quickstart

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — `pip install uv`
- Claude Code (OAuth auth for LLM calls — no API key needed) — or set `ANTHROPIC_API_KEY`

### Install

```bash
git clone https://github.com/Jimmycarroll2021/knowledgeforge
cd knowledgeforge
uv sync
```

### Full pipeline

```bash
# 1. Structural extraction — headings, links, tags (fast, no LLM)
knowledgeforge ingest --source /path/to/your/data

# 2. LLM semantic extraction — typed (subject, predicate, object) triples
knowledgeforge extract --source /path/to/your/data

# 3. Entity resolution — metaphone blocking → jaro+cosine → Union-Find → SAME_AS edges
knowledgeforge resolve

# 4. Embeddings — learned GraphSAGE aggregator + turbovec SIMD index
knowledgeforge embed

# 5. Communities — hierarchical Leiden detection + LLM summaries (enables global/drift query)
knowledgeforge community

# 6a. Local query — k-hop subgraph, grounded + cited
knowledgeforge query "what is GraphSAGE?"

# 6b. Global query — synthesise across community summaries (Edge et al. 2024)
knowledgeforge query "what are the main method families?" --mode global

# 6c. DRIFT-style query — community themes + local retrieval fusion (not the full Microsoft DRIFT loop)
knowledgeforge query "how does GraphSAGE relate to GCN?" --mode drift
```

### Other commands

```bash
knowledgeforge stats                                              # graph statistics by layer + predicate
knowledgeforge similar GraphSAGE                                  # fast SIMD similarity search
knowledgeforge similar "inductive node embedding" --text         # free-text similarity
knowledgeforge query x --path-only --from-entity A --to-entity B # shortest graph path (no LLM)
knowledgeforge provenance GraphSAGE                              # full provenance for an entity
knowledgeforge serve                                             # REST API on localhost:8000
```

---

## REST API

```bash
knowledgeforge serve          # localhost:8000
# or
docker-compose up             # with persistent named volume
```

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Status, entity/triple/embedding counts |
| `POST` | `/ingest` | Ingest a source directory |
| `POST` | `/query` | GraphRAG answer + `cited_triples`; `mode` = `local` or `global` |
| `POST` | `/embed` | Build/update embeddings |
| `GET`  | `/similar/{entity}` | Semantic + structural similarity search |
| `GET`  | `/graph/stats` | Graph statistics by layer and predicate |
| `GET`  | `/graph/node/{id}` | Entity + all its triples |
| `GET`  | `/graph/path?from=X&to=Y` | Shortest graph path |
| `GET`  | `/graph/provenance/{id}` | Full provenance chain |
| `POST` | `/graph/resolve` | Run entity resolution; returns alias stats |
| `POST` | `/graph/community` | Detect + summarise communities |
| `GET`  | `/graph/community` | List community summaries + stats |

Every triple in a response carries: `source_doc`, `layer`, `confidence`, `extraction_method`, `timestamp`, `lineage`.

### Local vs global query

```bash
# Local — specific entity questions, k-hop subgraph, grounds only on trusted layers
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "what is GraphSAGE?", "hops": 2, "mode": "local"}'

# Global — thematic questions, synthesises across community summaries (run /graph/community first)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "what are the major method families?", "mode": "global"}'
```

```json
{
  "question": "what is GraphSAGE?",
  "answer": "GraphSAGE is an inductive node embedding algorithm proposed by Hamilton...",
  "anchor_entities": ["GraphSAGE", "Hamilton 2017 - GraphSAGE"],
  "subgraph_size": 47,
  "mode": "local",
  "cited_triples": [
    {
      "subject": "GraphSAGE",
      "predicate": "PROPOSED_BY",
      "object": "Hamilton",
      "confidence": 0.9,
      "source_doc": "GraphSAGE.md",
      "layer": "source_facts",
      "evidence": "Hamilton, Ying & Leskovec (2017) introduced inductive node embedding",
      "lineage": []
    }
  ]
}
```

Local-mode grounding excludes the `llm_hypotheses` layer, so a speculative LLM-authored triple can never back a "grounded" answer.

---

## Security & configuration

Production hardening is **config-gated and OFF by default** (so the dev experience and tests are unchanged). Implemented in [`src/knowledgeforge/api/security.py`](src/knowledgeforge/api/security.py):

| Variable | Default | Behaviour |
|----------|---------|-----------|
| `KF_API_KEY` | unset (open) | When set, every request (except `/health`, `/docs`, `/redoc`, `/openapi.json`) must send header `X-API-Key: <value>` or gets `401`. |
| `KF_RATE_LIMIT` | unset (off) | When set to an integer, sliding-window limit of N requests/minute per client IP; over-limit returns `429` with `Retry-After`. `/health` is exempt. |
| `KF_CORS_ORIGINS` | `*` | Comma-separated allowed CORS origins. Restrict in production. |
| `KF_DB_PATH` | `data/graph.db` | SQLite graph store path. |
| `KF_EMBEDDINGS_PATH` | `data/embeddings` | ChromaDB embeddings store path. |
| `KF_MODEL` | `claude-haiku-4-5-20251001` | Default LLM model. |
| `ANTHROPIC_API_KEY` | — | Optional — Claude Code OAuth works without it. |

Structured JSON request logging and `X-Request-ID` propagation are always on (stdlib only, cheap). Env vars are read per-request, so auth and rate limiting can be toggled without rebuilding the app.

```bash
cp .env.example .env
export KF_API_KEY=$(openssl rand -hex 32)   # enable auth
export KF_RATE_LIMIT=120                      # 120 req/min/IP
knowledgeforge serve
```

See [`docs/production-checklist.md`](docs/production-checklist.md) for the full deployment checklist.

---

## Domain adapters

Domain swap = one `Adapter` class. The engine is unchanged.

```python
from knowledgeforge.contracts import Adapter, AdapterSchema, SourceDocument, Triple
from pathlib import Path

class MyAdapter:
    def scan(self, source: Path) -> list[SourceDocument]: ...
    def extract(self, doc: SourceDocument) -> list[Triple]: ...
    def schema(self) -> AdapterSchema: ...
```

Built-in adapters:
- `universal` — any file type (PDF, DOCX, HTML, CSV, JSON, Markdown, code)
- `vault` — Obsidian-style markdown vaults (wiki links, headings, tags)

An `AdapterSchema` may carry SHACL-style `PropertyShape` constraints (allowed target kinds, cardinality, datatype, severity). The pipeline enforces them on a severity ladder — soft by default (admit + flag), strict on opt-in (`ForgePipeline.run(..., strict=True)` rejects `Violation`-severity triples).

---

## Development

```bash
uv sync --dev
uv run pytest tests/ -q          # full suite green, incl. resolution-eval + security tests
uv run ruff check src/ tests/
```

---

## Research foundation

Every layer implements what the primary literature specifies — not a wrapper around it.

| Layer | Paper | What is actually implemented |
|-------|-------|------------------------------|
| Triple Primitive | W3C RDF 1.1 (2014) | `(subject, predicate, object)` graph model |
| Provenance | W3C PROV-O (2013) | `source` (producing agent), `lineage` (`wasDerivedFrom`), `valid_from`/`valid_to`, `schema_version` on every triple |
| Validation | W3C SHACL (2017) | `PropertyShape` cardinality / target-class / datatype + severity ladder |
| Entity Resolution | Winkler 1990; Union-Find WCC | metaphone blocking, jaro+cosine, 0.90 semantic floor, transitive Union-Find — **measured F1 = 1.00** |
| Node Embeddings | Hamilton et al. 2017 — GraphSAGE (NeurIPS) | **learned** unsupervised aggregator `z = L2(ReLU(W·CONCAT(self, mean_nbrs)))`, graph loss + negative sampling, hand-derived backprop |
| Similarity Search | Google TurboQuant (2024) | 4-bit SIMD quantised ANN via turbovec |
| Communities | Edge et al. 2024 — arXiv:2404.16130 | hierarchical Leiden detection + LLM community summaries |
| Graph Retrieval | Edge et al. 2024 — arXiv:2404.16130 | local k-hop (trusted-layer grounding) + global community-summary synthesis (rank top-K → single grounded synthesis; true map-reduce is roadmap) + DRIFT-style fusion |
| GNN Theory | Battaglia et al. 2018 — arXiv:1806.01261 | triple-as-unit + adapter-as-generalisation inductive bias |

Full mapping: [`docs/research-alignment.md`](docs/research-alignment.md)

---

## Roadmap (not yet built)

These are honestly **deferred** — the current single-machine engine does not do them yet:

- **Neo4j / GDS backend tier** for millions of entities (replaces the per-build N×N GraphSAGE matrix and O(n²) blocking, which are fine for hundreds–low-thousands of entities only).
- **Level-aware global search + community-report measurement (MRR@10)**. Hierarchical Leiden community detection (deterministic; `level` + `parent_community_id`, coarse→fine) and a **DRIFT** query mode are now built (`community/detector.py`, `inference/graphrag.py`); making global search drill the hierarchy and proving the gain with MRR@10 is the remaining work. Also pending: **true map-reduce** global synthesis (today it ranks top-K summaries and makes a single LLM call), the **full DRIFT loop** (primer → follow-up subqueries → refine; today it is themes + local fusion), and an **answer-quality eval harness** (retrieval MRR/nDCG + answer faithfulness).
- **TransE relation-as-first-class-vector scorer** (currently TransE is cited as the theoretical grounding for the triple unit, not implemented as a trained scorer).
- **Inductive new-node embedding path** (embed an unseen node without a full rebuild).
- **Pluggable adapter registry** (currently adapters are wired explicitly).

---

## License

MIT
