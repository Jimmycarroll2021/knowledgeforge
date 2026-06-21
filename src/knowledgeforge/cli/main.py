"""knowledgeforge CLI — ingest raw data, query the graph."""
from __future__ import annotations

import json
from pathlib import Path

import click

from ..adapter.vault import VaultAdapter
from ..adapter.universal import UniversalAdapter
from ..pipeline import ForgePipeline
from ..store.sqlite import SQLiteGraphStore

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
