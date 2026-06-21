# KnowledgeForge

> Raw unstructured data in → validated, queryable, provenance-backed knowledge graph out.

One command. No manual schema. No vendor lock-in. Domain swap = adapter swap only.

[![CI](https://github.com/Jimmycarroll2021/knowledgeforge/actions/workflows/ci.yml/badge.svg)](https://github.com/Jimmycarroll2021/knowledgeforge/actions/workflows/ci.yml)

---

## What it does

Drop any directory of files (Markdown, PDF, DOCX, HTML, CSV, JSON, code) into the pipeline. KnowledgeForge extracts structured knowledge as typed triples with full provenance, resolves entities, builds semantic + structural embeddings, and answers graph-aware questions grounded in cited evidence.

```bash
knowledgeforge ingest --source ~/my-notes
knowledgeforge extract --source ~/my-notes
knowledgeforge query "what connects GraphSAGE to entity resolution?"
```

```
GraphRAG query (2-hop)

Answer:
GraphSAGE [PROPOSED_BY Hamilton] uses neighbourhood aggregation
[TYPE_OF inductive node embedding] which [RELATED_TO entity representations]
used in resolution...

Top evidence:
  (GraphSAGE)-[PROPOSED_BY]->(Hamilton)  [conf=0.90]  source: GraphSAGE.md
  (GraphSAGE)-[TYPE_OF]->(inductive node embedding)  [conf=0.85]
```

---

## Architecture

```
Raw files (any format)
    ↓
[Adapter]            ← domain swap = adapter swap only; core engine unchanged
    ↓
[Triple Primitive]   ← (subject, predicate, object) + provenance (PROV-O)
    ↓
[Entity Resolution]  ← 3-phase: exact → Jaro-Winkler 0.85 → WCC Union-Find
    ↓
[Embedding Pipeline] ← sentence-transformers + GraphSAGE MEAN agg + turbovec SIMD
    ↓
[Graph Store]        ← SQLite WAL, 4 layers: source_facts/normalised/inferred/llm
    ↓
[GraphRAG]           ← k-hop BFS subgraph → grounded LLM answer + cited_triples
    ↓
[REST API]           ← FastAPI, provenance on every response
```

Every design decision maps to primary literature — see [`docs/research-alignment.md`](docs/research-alignment.md).

---

## Quickstart

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — `pip install uv`
- Claude Code (OAuth auth for LLM calls, no API key needed) — or set `ANTHROPIC_API_KEY`

### Install

```bash
git clone https://github.com/Jimmycarroll2021/knowledgeforge
cd knowledgeforge
uv sync
```

### Ingest your data

```bash
# Structural extraction — headings, links, tags (fast, no LLM)
knowledgeforge ingest --source /path/to/your/data

# LLM semantic extraction — typed (subject, predicate, object) triples
knowledgeforge extract --source /path/to/your/data

# Check the graph
knowledgeforge stats
```

### Resolve + embed

```bash
knowledgeforge resolve   # 3-phase entity resolution
knowledgeforge embed     # GraphSAGE + turbovec SIMD index
```

### Query

```bash
# Graph-aware question answering (GraphRAG)
knowledgeforge query "what is GraphSAGE?"

# Semantic similarity
knowledgeforge similar GraphSAGE

# Shortest graph path between two entities
knowledgeforge query x --path-only --from-entity GraphSAGE --to-entity TransE

# Full provenance for an entity
knowledgeforge provenance GraphSAGE
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
| `GET` | `/health` | Status, entity/triple/embedding counts |
| `POST` | `/ingest` | Ingest a source directory |
| `POST` | `/query` | GraphRAG answer + `cited_triples` with provenance |
| `POST` | `/embed` | Build/update embeddings |
| `GET` | `/graph/stats` | Graph statistics by layer and predicate |
| `GET` | `/graph/node/{id}` | Entity + all triples |
| `GET` | `/graph/provenance/{id}` | Full provenance chain |
| `GET` | `/graph/path?from=X&to=Y` | Shortest graph path |
| `GET` | `/similar/{entity}` | Semantic + structural similarity search |

Every response carries: `source_doc`, `layer`, `confidence`, `extraction_method`, `timestamp`.

### Example query response

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "what is GraphSAGE?", "hops": 2}'
```

```json
{
  "question": "what is GraphSAGE?",
  "answer": "GraphSAGE is an inductive node embedding algorithm proposed by Hamilton...",
  "anchor_entities": ["GraphSAGE", "Hamilton 2017 - GraphSAGE"],
  "subgraph_size": 47,
  "cited_triples": [
    {
      "subject": "GraphSAGE",
      "predicate": "PROPOSED_BY",
      "object": "Hamilton",
      "confidence": 0.9,
      "source_doc": "GraphSAGE.md",
      "layer": "source_facts",
      "evidence": "Hamilton, Ying & Leskovec (2017) introduced inductive node embedding"
    }
  ]
}
```

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

---

## Configuration

```bash
cp .env.example .env
# Set ANTHROPIC_API_KEY (optional — Claude Code OAuth works without it)
```

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Anthropic API key (optional if Claude Code installed) |
| `KF_DB_PATH` | `data/graph.db` | SQLite graph store path |
| `KF_EMBEDDINGS_PATH` | `data/embeddings` | ChromaDB embeddings store |
| `KF_CORS_ORIGINS` | `*` | Comma-separated allowed CORS origins |

---

## Development

```bash
uv sync --dev
uv run pytest tests/ -q
uv run ruff check src/ tests/
```

---

## Research foundation

| Layer | Paper |
|-------|-------|
| Triple Primitive | W3C RDF 1.1 (2014) |
| Provenance | W3C PROV-O (2013) |
| Entity Resolution | Jaro-Winkler 1990, Union-Find WCC |
| Node Embeddings | Hamilton et al. 2017 — GraphSAGE (NeurIPS) |
| Relation Embeddings | Bordes et al. 2013 — TransE (NeurIPS) |
| Graph Retrieval | Edge et al. 2024 — arXiv:2404.16130 |
| GNN Theory | Battaglia et al. 2018 — arXiv:1806.01261 |

Full mapping: [`docs/research-alignment.md`](docs/research-alignment.md)

---

## License

MIT
