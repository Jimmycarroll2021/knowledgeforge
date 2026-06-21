# KnowledgeForge

> Raw unstructured data in → validated, queryable, provenance-backed knowledge graph out.

Clone the repo. Drop your data. Run the pipeline. Get a best-in-class knowledge graph.

Swapping the domain **Adapter** is the only change needed for a new data source. The core engine is frozen.

---

## Architecture

```
Raw Data
    ↓
[ Adapter Layer ]          — domain-specific schema → canonical Triple contract
    ↓
[ Triple Primitive ]       — S→P→O + evidence + confidence + full PROV-O provenance
    ↓
[ SHACL Validation ]       — shape validation before any graph commit
    ↓
[ Entity Resolution ]      — blocking → embedding similarity → merge policy
    ↓
[ Graph Store ]            — SQLite (dev), pluggable (prod); fact/inference layers separated
    ↓
[ Embedding Pipeline ]     — GraphSAGE-style node embeddings + ChromaDB
    ↓
[ GraphRAG Inference ]     — graph-aware retrieval (entities, paths, communities) + LLM grounding
    ↓
[ Query API ]              — REST: ingest, query, graph traversal, provenance lookup
```

## Research Alignment

Built from primary literature:
- Hamilton et al. (2017) — GraphSAGE
- Bordes et al. (2013) — TransE  
- Edge et al. (2024) — Microsoft GraphRAG (arXiv:2404.16130)
- W3C RDF 1.1, SHACL, PROV-O

## Quickstart

```bash
git clone https://github.com/Jimmycarroll2021/knowledgeforge
cd knowledgeforge
cp config.example.yaml config.yaml   # set ANTHROPIC_API_KEY
docker-compose up
# → API at localhost:8000
```

Or with uv (local dev):

```bash
uv sync
uv run knowledgeforge ingest --adapter vault --source /path/to/vault --dry-run
uv run knowledgeforge query "what connects GraphSAGE to entity resolution?"
```

## Status

**Phase 1 — Foundation** (in progress)

See `.planning/ROADMAP.md` for full 7-phase plan.
