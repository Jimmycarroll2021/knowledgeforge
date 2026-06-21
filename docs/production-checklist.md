# Production Checklist

Checklist for deploying KnowledgeForge outside a dev laptop. Scope is **single-machine**
(hundreds–low-thousands of entities); a multi-node backend is on the roadmap, not built.

---

## Secrets & Credentials

- [ ] `ANTHROPIC_API_KEY` set via environment (optional — Claude Code OAuth works without it; never hardcode)
- [ ] No `.env` committed (`.gitignore` covers it — verify with `git ls-files | grep .env`)
- [ ] `KF_API_KEY` set to a strong random value to enable API auth (see API section) — store via Docker secrets / vault in production

## Data

- [ ] SQLite WAL mode confirmed (`PRAGMA journal_mode=WAL` — set in `store/sqlite.py` DDL)
- [ ] `data/` is on a persistent volume (not ephemeral container storage)
- [ ] Backups: `cp data/graph.db data/graph.db.bak` before any bulk ingest
- [ ] ChromaDB embeddings path (and cached `graphsage_w.npy`) on the same persistent volume

## API — built-in, config-gated hardening

Hardening lives in `src/knowledgeforge/api/security.py`. It is **OFF by default** so dev and tests
are unchanged; enable it via environment variables for production. Env vars are read per-request, so
toggling needs no rebuild.

- [ ] **Auth:** set `KF_API_KEY` → every request (except `/health`, `/docs`, `/redoc`,
      `/openapi.json`) must send `X-API-Key: <value>` or gets `401`. The `/ingest` endpoint accepts
      arbitrary file paths — do not expose it without `KF_API_KEY`.
- [ ] **Rate limiting:** set `KF_RATE_LIMIT=<N>` → sliding-window N requests/minute per client IP;
      over-limit returns `429` with `Retry-After`. `/health` is exempt. (An upstream nginx/Caddy
      limiter can still be layered on if you need cross-process limits — the built-in limiter is
      per-process and in-memory.)
- [ ] **CORS:** set `KF_CORS_ORIGINS` to known origins (not the `*` default).
- [ ] `uvicorn --workers N` for multi-process serving (N = CPU cores). Note: the in-memory rate
      limiter counts per worker process — front with a shared limiter if exact global limits matter.

## Observability — built in

- [ ] Structured JSON request logging is always on (stdlib): one line per request with
      `request_id`, `method`, `path`, `status`, `duration_ms`, `client`.
- [ ] `X-Request-ID` is assigned (or propagated) on every response for trace correlation.
- [ ] Uvicorn access logs enabled (`--access-log`) if you also want server-level logs.
- [ ] Health check wired: `GET /health` returns 200 with entity/triple/embedding counts.
- [ ] Docker `HEALTHCHECK` configured in `docker-compose.yml`.
- [ ] Alert if `/health` entity or triple counts drop significantly between runs.

## CI/CD

- [ ] GitHub Actions CI passes (lint + full test suite, incl. resolution-eval + security tests) on every push
- [ ] Docker image built and smoke-tested in CI before deploy
- [ ] Pinned Python 3.12 and `uv.lock` committed

## Performance (single-machine scope)

- [ ] Embeddings prebuilt before first query (`knowledgeforge embed`)
- [ ] Communities prebuilt before global queries (`knowledgeforge community`)
- [ ] turbovec SIMD index loaded at startup (done in the API lifespan)
- [ ] SQLite indexes on `subject`, `object`, `predicate`, `layer`, `source_doc` (in schema)
- [ ] Note the scale ceiling: the per-build N×N GraphSAGE training matrix and O(n²) candidate
      blocking are fine for hundreds–low-thousands of entities. For millions, the Neo4j/GDS backend
      tier (roadmap, **not yet built**) is required — do not point this build at a million-node graph.

## Data Governance

- [ ] Full PROV-O provenance on every triple: `source` (producing agent), `lineage`, `valid_from`/
      `valid_to`, `schema_version` — in addition to `source_doc`, `extraction_method`, `timestamp`
- [ ] Named layers separate source facts from inference and LLM hypotheses; local-mode GraphRAG
      grounds only on trusted layers (`llm_hypotheses` excluded)
- [ ] Entity-alias table + `SAME_AS` edges preserve originals — resolution is non-destructive
- [ ] SHACL-style validation available: enable `ForgePipeline.run(..., strict=True)` to reject
      `Violation`-severity triples at ingest
- [ ] Audit log: `git log` on the repo captures when data was ingested
