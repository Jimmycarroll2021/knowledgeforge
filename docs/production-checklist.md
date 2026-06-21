# Production Checklist

Checklist for deploying KnowledgeForge outside a dev laptop.

---

## Secrets & Credentials

- [ ] `ANTHROPIC_API_KEY` set via environment variable (never hardcoded)
- [ ] No `.env` file committed (`.gitignore` covers it — verify with `git ls-files | grep .env`)
- [ ] Docker secrets or vault (e.g. AWS Secrets Manager) for production key storage

## Data

- [ ] SQLite WAL mode confirmed (`PRAGMA journal_mode=WAL` — default in code)
- [ ] `data/` directory is on a persistent volume (not ephemeral container storage)
- [ ] Backups: `cp data/graph.db data/graph.db.bak` before any bulk ingest
- [ ] ChromaDB embeddings path is on the same persistent volume

## API

- [ ] `KF_CORS_ORIGINS` restricted to known origins (not `*`)
- [ ] Rate limiting in front of the API (nginx, Caddy, or a gateway)
- [ ] `/ingest` endpoint should be behind auth — it accepts arbitrary file paths
- [ ] `uvicorn --workers N` for multi-process serving (N = CPU cores)

## Observability

- [ ] Uvicorn access logs enabled (`--access-log`)
- [ ] Health check wired: `GET /health` returns 200 with entity/triple counts
- [ ] Docker `HEALTHCHECK` is configured in `docker-compose.yml` (already present)
- [ ] Alert if `/health` entities or triples drop significantly between runs

## CI/CD

- [ ] GitHub Actions CI passes (lint + test) on every push
- [ ] Docker image built and smoke-tested in CI before deploy
- [ ] Pinned Python version (3.12) and uv lockfile committed (`uv.lock`)

## Performance

- [ ] Embeddings prebuilt before first query (`knowledgeforge embed`)
- [ ] turbovec SIMD index loaded at startup (done in lifespan)
- [ ] SQLite indexes on `subject`, `object`, `predicate`, `layer` (already in schema)
- [ ] For high-query load: consider read-only replica of `graph.db`

## Data Governance

- [ ] Source provenance on every triple: `source_doc`, `extraction_method`, `timestamp`
- [ ] `layer` field separates source facts from LLM hypotheses — never mix in queries
- [ ] Entity aliases table preserves original entities — resolution is non-destructive
- [ ] Audit log: `git log` on the repo captures when data was ingested
