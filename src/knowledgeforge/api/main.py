"""KnowledgeForge FastAPI application.

Lifespan: initialises SQLiteGraphStore + EmbeddingPipeline once on startup,
shared via app.state across all requests.

Environment variables (all optional):
  KF_DB_PATH         — path to SQLite graph store (default: data/graph.db)
  KF_EMBEDDINGS_PATH — path to ChromaDB embeddings store (default: data/embeddings)
  KF_MODEL           — default LLM model (default: claude-haiku-4-5-20251001)
  KF_CORS_ORIGINS    — comma-separated CORS origins (default: *)
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..store.sqlite import SQLiteGraphStore
from ..embeddings.pipeline import EmbeddingPipeline
from .routes import health, ingest, query, embed, graph


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_path = Path(_env("KF_DB_PATH", "data/graph.db"))
    embeddings_path = _env("KF_EMBEDDINGS_PATH", "data/embeddings")

    store = SQLiteGraphStore(db_path)
    embed_pipeline = EmbeddingPipeline(store, chroma_path=embeddings_path)
    embed_pipeline._build_turbo_index()

    app.state.store = store
    app.state.embed = embed_pipeline
    app.state.db_path = str(db_path)
    app.state.embeddings_path = embeddings_path

    yield

    store.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="KnowledgeForge",
        description="Raw unstructured data in → validated, queryable, provenance-backed knowledge graph out.",
        version="0.1.0",
        lifespan=lifespan,
    )

    origins = _env("KF_CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, tags=["ops"])
    app.include_router(ingest.router, tags=["ingest"])
    app.include_router(query.router, tags=["query"])
    app.include_router(embed.router, tags=["embed"])
    app.include_router(graph.router, tags=["graph"])

    return app


app = create_app()
