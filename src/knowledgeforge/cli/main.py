"""knowledgeforge CLI — ingest raw data, query the graph."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click

# Windows cp1252 terminals can't handle Unicode → force UTF-8 output
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from ..adapter.vault import VaultAdapter
from ..adapter.universal import UniversalAdapter
from ..pipeline import ForgePipeline
from ..store.sqlite import SQLiteGraphStore


def _load_env() -> None:
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

_ADAPTERS = {
    "vault": VaultAdapter,
    "universal": UniversalAdapter,
}

_DEFAULT_DB = Path("data/graph.db")


@click.group()
def cli() -> None:
    """KnowledgeForge — raw data in, best-in-class knowledge graph out."""


@cli.command()
@click.option("--adapter", "-a", default="universal",
              type=click.Choice(list(_ADAPTERS)), show_default=True,
              help="Adapter to use. 'universal' handles any file type.")
@click.option("--source", "-s", required=True,
              type=click.Path(exists=True, path_type=Path),
              help="Path to your raw data — any file or directory.")
@click.option("--db", default=str(_DEFAULT_DB), show_default=True,
              help="SQLite graph store path.")
@click.option("--include-dirs", multiple=True,
              help="Only ingest these top-level directories (vault adapter). Repeat for multiple.")
@click.option("--dry-run", is_flag=True,
              help="Scan without writing — shows what would be ingested.")
@click.option("--max-facts", default=40, show_default=True,
              help="Max triples extracted per file.")
def ingest(
    adapter: str,
    source: Path,
    db: str,
    include_dirs: tuple[str, ...],
    dry_run: bool,
    max_facts: int,
) -> None:
    """Ingest raw data from SOURCE into the knowledge graph.

    \b
    Examples:
      knowledgeforge ingest --source ~/KnowledgeGraph --dry-run
      knowledgeforge ingest --source ~/KnowledgeGraph --include-dirs algorithms --include-dirs concepts
      knowledgeforge ingest --adapter vault --source /path/to/vault
    """
    adapter_cls = _ADAPTERS[adapter]
    adapter_kwargs: dict = {"max_facts_per_file": max_facts}
    if include_dirs:
        adapter_kwargs["include_dirs"] = set(include_dirs)

    adp = adapter_cls(**adapter_kwargs)
    store = SQLiteGraphStore(Path(db))
    pipeline = ForgePipeline(store)

    label = "[DRY RUN] " if dry_run else ""
    click.echo(f"\n{label}KnowledgeForge — ingesting via {adapter!r} adapter")
    click.echo(f"  source:  {source}")
    click.echo(f"  store:   {db}\n")

    result = pipeline.run(adp, source, dry_run=dry_run)

    click.echo(result.summary())

    if not dry_run:
        stats = store.stats()
        click.echo(f"\nGraph stats:")
        click.echo(f"  entities: {stats['entities']}")
        click.echo(f"  triples:  {stats['triples']}")
        if stats["by_predicate"]:
            click.echo(f"\n  Top predicates:")
            for pred, count in list(stats["by_predicate"].items())[:8]:
                click.echo(f"    {pred:<30} {count}")

    store.close()

    if result.errors:
        raise SystemExit(1)


@cli.command()
@click.option("--db", default=str(_DEFAULT_DB), show_default=True,
              help="SQLite graph store path.")
def stats(db: str) -> None:
    """Show graph store statistics."""
    db_path = Path(db)
    if not db_path.exists():
        click.echo(f"No graph store at {db} — run 'knowledgeforge ingest' first.")
        raise SystemExit(1)

    store = SQLiteGraphStore(db_path)
    s = store.stats()
    store.close()

    click.echo(f"\nKnowledgeForge graph — {db}")
    click.echo(f"  entities: {s['entities']}")
    click.echo(f"  triples:  {s['triples']}")
    click.echo(f"\n  By layer:")
    for layer, count in s.get("by_layer", {}).items():
        click.echo(f"    {layer:<30} {count}")
    click.echo(f"\n  Top predicates:")
    for pred, count in list(s.get("by_predicate", {}).items())[:10]:
        click.echo(f"    {pred:<30} {count}")


@cli.command()
@click.argument("subject")
@click.option("--db", default=str(_DEFAULT_DB), show_default=True)
@click.option("--fmt", default="text", type=click.Choice(["text", "json"]))
def provenance(subject: str, db: str, fmt: str) -> None:
    """Show all triples for SUBJECT (entity id or doc path fragment)."""
    store = SQLiteGraphStore(Path(db))
    rows = store.provenance(subject)
    store.close()

    if not rows:
        click.echo(f"No triples found for: {subject}")
        raise SystemExit(1)

    if fmt == "json":
        click.echo(json.dumps(rows, indent=2))
        return

    for r in rows:
        click.echo(
            f"  {r['subject'][:40]:<42} {r['predicate']:<25} {r['object'][:50]}"
            f"  [{r['confidence']:.2f}] via {r['source_doc']}"
        )


@cli.command()
@click.option("--source", "-s", required=True,
              type=click.Path(exists=True, path_type=Path),
              help="Source directory or file to extract triples from.")
@click.option("--db", default=str(_DEFAULT_DB), show_default=True)
@click.option("--model", default="claude-haiku-4-5-20251001", show_default=True,
              help="Claude model for extraction.")
@click.option("--limit", default=None, type=int,
              help="Max files to process (useful for testing).")
@click.option("--dry-run", is_flag=True, help="Show what would be extracted, don't write.")
def extract(source: Path, db: str, model: str, limit: int | None, dry_run: bool) -> None:
    """Run LLM semantic triple extraction on SOURCE.

    Reads each section of each file and asks Claude to extract
    factual (subject, predicate, object) triples. Stores in
    source_facts layer with extraction_method='llm'.

    \b
    Requires: ANTHROPIC_API_KEY in .env or environment.

    \b
    Examples:
      knowledgeforge extract --source ~/KnowledgeGraph/algorithms --limit 5
      knowledgeforge extract --source ~/KnowledgeGraph --db data/graph.db
    """
    _load_env()
    from ..adapter.universal import UniversalAdapter
    from ..extractor.llm import LLMExtractor
    from ..store.sqlite import SQLiteGraphStore

    adp = UniversalAdapter()
    docs = adp.scan(source)
    if limit:
        docs = docs[:limit]

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    backend = "Anthropic SDK" if api_key else "claude CLI (OAuth)"
    click.echo(f"\nLLM extraction — {len(docs)} files via {model} [{backend}]")
    click.echo(f"  source: {source}")
    if dry_run:
        click.echo("  [DRY RUN — no writes]")
        for d in docs:
            click.echo(f"    {d.relative_path}")
        return

    extractor = LLMExtractor(model=model)
    store = SQLiteGraphStore(Path(db))

    total_added = total_skipped = total_triples = 0
    with click.progressbar(docs, label="Extracting") as bar:
        for doc in bar:
            triples = extractor.extract(doc)
            total_triples += len(triples)
            added, skipped = store.add_triples(triples)
            total_added += added
            total_skipped += skipped

    store.close()
    click.echo(f"\nLLM extraction complete:")
    click.echo(f"  files processed:  {len(docs)}")
    click.echo(f"  triples extracted:{total_triples}")
    click.echo(f"  triples added:    {total_added}")
    click.echo(f"  duplicates skipped:{total_skipped}")


@cli.command()
@click.option("--db", default=str(_DEFAULT_DB), show_default=True)
@click.option("--threshold", default=0.85, show_default=True,
              help="Jaro-Winkler similarity threshold for Phase 2 match.")
def resolve(db: str, threshold: float) -> None:
    """Run entity resolution — 3-phase pipeline from the vault spec.

    Phase 1: exact normalised name match (same type)
    Phase 2: Jaro-Winkler >= threshold (same type, blocked)
    Phase 3: WCC on SIMILAR_TO edges

    Writes SAME_AS edges to entity_aliases table (originals kept for provenance).
    """
    store = SQLiteGraphStore(Path(db))
    from ..resolution.resolver import EntityResolver
    resolver = EntityResolver(store, threshold=threshold)

    click.echo(f"\nEntity resolution (threshold={threshold})")
    result = resolver.resolve()
    store.close()

    click.echo(f"  Phase 1 (exact):       {result['phase1_merges']} merges")
    click.echo(f"  Phase 2 (jaro-winkler):{result['phase2_merges']} merges")
    click.echo(f"  Phase 3 (structural):  {result['phase3_merges']} merges")
    click.echo(f"  Total aliases written: {result['total_merges']}")


@cli.command()
@click.argument("question")
@click.option("--db", default=str(_DEFAULT_DB), show_default=True)
@click.option("--hops", default=2, show_default=True,
              help="k-hop neighbourhood expansion depth.")
@click.option("--model", default="claude-haiku-4-5-20251001", show_default=True)
@click.option("--path-only", is_flag=True, help="Find graph paths only — no LLM call.")
@click.option("--from-entity", default=None, help="Path mode: start entity.")
@click.option("--to-entity", default=None, help="Path mode: end entity.")
def query(
    question: str,
    db: str,
    hops: int,
    model: str,
    path_only: bool,
    from_entity: str | None,
    to_entity: str | None,
) -> None:
    """Query the graph using GraphRAG — graph-aware retrieval + LLM answer.

    Extracts anchor entities from QUESTION, expands k-hop subgraph,
    injects facts into Claude context, returns grounded answer.

    \b
    From concepts/GraphRAG.md:
      KGSWC 2024: hallucination F1 0.77 → 0.94 with KG injection.

    \b
    Requires: ANTHROPIC_API_KEY in .env or environment.

    \b
    Examples:
      knowledgeforge query "what is GraphSAGE and how does it work?"
      knowledgeforge query "how does entity resolution connect to GraphSAGE?"
      knowledgeforge query "x" --path-only --from-entity GraphSAGE --to-entity TransE
    """
    _load_env()
    from ..inference.graphrag import GraphRAG

    store = SQLiteGraphStore(Path(db))
    rag = GraphRAG(store, model=model, hops=hops)

    if path_only and from_entity and to_entity:
        paths = rag.path(from_entity, to_entity)
        if not paths:
            click.echo(f"No path found between {from_entity!r} and {to_entity!r}")
        else:
            for edge in paths[0]:
                click.echo(f"  ({edge['subject']})-[{edge['predicate']}]->({edge['object']})")
        store.close()
        return

    click.echo(f"\nGraphRAG query ({hops}-hop, model={model})")
    click.echo(f"  question: {question}\n")

    result = rag.ask(question)
    store.close()

    click.echo(f"Anchor entities: {', '.join(result['anchor_entities'][:5]) or 'none'}")
    click.echo(f"Subgraph size:   {result['subgraph_size']} triples\n")
    click.echo("Answer:")
    click.echo(result["answer"])

    if result["evidence"]:
        click.echo(f"\nTop evidence ({min(5, len(result['evidence']))} triples):")
        skip = {"CONTAINS_KEY", "HAS_FILE_TYPE"}
        shown = 0
        for t in result["evidence"]:
            if t["predicate"] not in skip and shown < 5:
                click.echo(f"  ({t['subject']})-[{t['predicate']}]->({t['object']})"
                           f"  [conf={t['confidence']:.2f}]")
                shown += 1


@cli.command()
@click.option("--db", default=str(_DEFAULT_DB), show_default=True)
@click.option("--embeddings", default="data/embeddings", show_default=True,
              help="ChromaDB embeddings store path.")
@click.option("--model", default="all-MiniLM-L6-v2", show_default=True,
              help="Sentence-transformer model for semantic embeddings.")
@click.option("--batch-size", default=64, show_default=True)
def embed(db: str, embeddings: str, model: str, batch_size: int) -> None:
    """Build entity embeddings — semantic + GraphSAGE-style structural aggregation.

    Uses sentence-transformers for semantic embeddings, mean-pools
    1-hop neighbour embeddings (GraphSAGE MEAN aggregator), stores
    in ChromaDB, builds turbovec SIMD index for fast similarity search.

    \b
    From ROADMAP.md Phase 4 / Hamilton 2017 (GraphSAGE):
      h_v = MEAN(h_v ∪ {h_u ∀u ∈ N(v)})

    \b
    Examples:
      knowledgeforge embed
      knowledgeforge embed --model all-mpnet-base-v2 --batch-size 32
    """
    store = SQLiteGraphStore(Path(db))
    from ..embeddings.pipeline import EmbeddingPipeline

    pipeline = EmbeddingPipeline(store, chroma_path=embeddings, model_name=model)

    click.echo(f"\nEmbedding pipeline — {model}")
    click.echo(f"  graph:      {db}")
    click.echo(f"  embeddings: {embeddings}\n")

    result = pipeline.embed_all(batch_size=batch_size)
    store.close()

    click.echo(f"  entities embedded: {result['entities_embedded']}")
    click.echo(f"  already cached:    {result['skipped']}")
    click.echo(f"  model:             {result['model']}")


@cli.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8000, show_default=True)
@click.option("--db", default=str(_DEFAULT_DB), show_default=True)
@click.option("--embeddings", default="data/embeddings", show_default=True)
@click.option("--reload", is_flag=True, help="Auto-reload on code changes (dev only).")
def serve(host: str, port: int, db: str, embeddings: str, reload: bool) -> None:
    """Start the KnowledgeForge REST API server.

    \b
    Examples:
      knowledgeforge serve
      knowledgeforge serve --host 127.0.0.1 --port 8080
      knowledgeforge serve --reload
    """
    import uvicorn
    os.environ.setdefault("KF_DB_PATH", db)
    os.environ.setdefault("KF_EMBEDDINGS_PATH", embeddings)
    uvicorn.run(
        "knowledgeforge.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


@cli.command("similar")
@click.argument("entity")
@click.option("--db", default=str(_DEFAULT_DB), show_default=True)
@click.option("--embeddings", default="data/embeddings", show_default=True)
@click.option("-k", default=10, show_default=True, help="Number of results.")
@click.option("--kind", default=None, help="Filter by entity kind (Algorithm, Concept, etc.)")
@click.option("--text", is_flag=True, help="Treat ENTITY as a free-text query instead of entity ID.")
def similar(entity: str, db: str, embeddings: str, k: int, kind: str | None, text: bool) -> None:
    """Find entities similar to ENTITY using semantic + structural embeddings.

    Uses turbovec (4-bit SIMD quantisation) for fast approximate search.

    \b
    Examples:
      knowledgeforge similar GraphSAGE
      knowledgeforge similar GraphSAGE --kind Algorithm -k 5
      knowledgeforge similar "inductive node embedding" --text
    """
    store = SQLiteGraphStore(Path(db))
    from ..embeddings.pipeline import EmbeddingPipeline

    pipeline = EmbeddingPipeline(store, chroma_path=embeddings)
    pipeline._build_turbo_index()

    if text:
        results = pipeline.search_by_text(entity, k=k, kind_filter=kind)
    else:
        results = pipeline.find_similar(entity, k=k, kind_filter=kind)
    store.close()

    if not results:
        click.echo(f"No embeddings found for '{entity}'. Run 'knowledgeforge embed' first.")
        return

    click.echo(f"\nTop {len(results)} similar to '{entity}':")
    for r in results:
        click.echo(f"  [{r['score']:.3f}] {r['entity_id'][:60]}  ({r['kind']})")
