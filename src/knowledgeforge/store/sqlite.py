"""SQLite graph store — the default GraphStore implementation.

Schema keeps four layers distinct: source_facts / normalised_triples /
inferred_relations / llm_hypotheses. Query by layer or across all layers.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ..contracts import Triple


_LAYERS = ("source_facts", "normalised_triples", "inferred_relations", "llm_hypotheses")

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS entities (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS triples (
    triple_id           TEXT PRIMARY KEY,
    subject             TEXT NOT NULL,
    predicate           TEXT NOT NULL,
    object              TEXT NOT NULL,
    source_kind         TEXT NOT NULL,
    target_kind         TEXT NOT NULL,
    evidence            TEXT NOT NULL,
    confidence          REAL NOT NULL,
    source_doc          TEXT NOT NULL,
    extraction_method   TEXT NOT NULL,
    timestamp           TEXT NOT NULL,
    adapter             TEXT NOT NULL,
    layer               TEXT NOT NULL DEFAULT 'source_facts',
    source              TEXT NOT NULL DEFAULT '',
    lineage             TEXT NOT NULL DEFAULT '[]',
    valid_from          TEXT,
    valid_to            TEXT,
    schema_version      TEXT NOT NULL DEFAULT '1',
    object_is_literal   INTEGER NOT NULL DEFAULT 0,
    object_datatype     TEXT,
    CHECK(confidence >= 0.0 AND confidence <= 1.0),
    CHECK(layer IN ('source_facts','normalised_triples','inferred_relations','llm_hypotheses'))
);

CREATE INDEX IF NOT EXISTS idx_t_subject   ON triples(subject);
CREATE INDEX IF NOT EXISTS idx_t_object    ON triples(object);
CREATE INDEX IF NOT EXISTS idx_t_predicate ON triples(predicate);
CREATE INDEX IF NOT EXISTS idx_t_layer     ON triples(layer);
CREATE INDEX IF NOT EXISTS idx_t_source    ON triples(source_doc);
"""

# PROV-O / RDF-literal columns added after the original schema shipped. Each is
# (name, "ddl") so a pre-existing DB can be brought forward via ALTER TABLE.
_MIGRATION_COLUMNS = [
    ("source", "TEXT NOT NULL DEFAULT ''"),
    ("lineage", "TEXT NOT NULL DEFAULT '[]'"),
    ("valid_from", "TEXT"),
    ("valid_to", "TEXT"),
    ("schema_version", "TEXT NOT NULL DEFAULT '1'"),
    ("object_is_literal", "INTEGER NOT NULL DEFAULT 0"),
    ("object_datatype", "TEXT"),
]


class SQLiteGraphStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._migrate()
        self._conn.commit()
        self._kind_conflicts: list[tuple[str, str, str]] = []

    def _migrate(self) -> None:
        """Bring a pre-existing DB forward by adding any missing PROV-O columns.

        Idempotent: introspects the live schema via PRAGMA table_info and only
        ALTERs columns that are absent, so it is a no-op on fresh DBs.
        """
        existing = {
            row[1] for row in self._conn.execute("PRAGMA table_info(triples)").fetchall()
        }
        for name, ddl in _MIGRATION_COLUMNS:
            if name not in existing:
                self._conn.execute(f"ALTER TABLE triples ADD COLUMN {name} {ddl}")

    def add_triple(self, triple: Triple) -> bool:
        """Insert a triple. Returns True if inserted, False if already exists."""
        tid = triple.triple_id()
        cur = self._conn.execute(
            "SELECT 1 FROM triples WHERE triple_id = ?", (tid,)
        )
        if cur.fetchone():
            return False

        with self._conn:
            for entity_id, kind in [
                (triple.subject, triple.source_kind),
                (triple.object, triple.target_kind),
            ]:
                existing = self._conn.execute(
                    "SELECT kind FROM entities WHERE id = ?", (entity_id,)
                ).fetchone()
                if existing is not None and existing[0] != kind:
                    self._kind_conflicts.append((entity_id, existing[0], kind))
                self._conn.execute(
                    "INSERT OR IGNORE INTO entities(id, kind, created_at) VALUES (?,?,?)",
                    (entity_id, kind, triple.timestamp),
                )
            self._conn.execute(
                """INSERT INTO triples(
                    triple_id, subject, predicate, object,
                    source_kind, target_kind, evidence, confidence,
                    source_doc, extraction_method, timestamp, adapter, layer,
                    source, lineage, valid_from, valid_to, schema_version,
                    object_is_literal, object_datatype
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    tid,
                    triple.subject, triple.predicate, triple.object,
                    triple.source_kind, triple.target_kind,
                    triple.evidence, triple.confidence,
                    triple.source_doc, triple.extraction_method,
                    triple.timestamp, triple.adapter, triple.layer,
                    triple.source, json.dumps(list(triple.lineage)),
                    triple.valid_from, triple.valid_to, triple.schema_version,
                    int(triple.object_is_literal), triple.object_datatype,
                ),
            )
        return True

    def kind_conflicts(self) -> list[tuple[str, str, str]]:
        """Entity-kind disagreements seen during inserts: (id, existing, new)."""
        return list(self._kind_conflicts)

    def add_triples(self, triples: list[Triple]) -> tuple[int, int]:
        """Bulk insert. Returns (added, skipped)."""
        added = skipped = 0
        for t in triples:
            if self.add_triple(t):
                added += 1
            else:
                skipped += 1
        return added, skipped

    def query(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
        layer: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if subject:
            clauses.append("subject = ?")
            params.append(subject)
        if predicate:
            clauses.append("predicate = ?")
            params.append(predicate)
        if obj:
            clauses.append("object = ?")
            params.append(obj)
        if layer:
            clauses.append("layer = ?")
            params.append(layer)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = self._conn.execute(
            f"SELECT * FROM triples {where} LIMIT ?", params
        ).fetchall()
        return [dict(r) for r in rows]

    def provenance(self, subject: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """SELECT t.*, e.kind as subject_kind
               FROM triples t JOIN entities e ON t.subject = e.id
               WHERE t.subject = ? OR t.object = ?
               ORDER BY t.timestamp""",
            (subject, subject),
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        entity_count = self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        triple_count = self._conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
        by_layer = {
            row[0]: row[1]
            for row in self._conn.execute(
                "SELECT layer, COUNT(*) FROM triples GROUP BY layer"
            ).fetchall()
        }
        by_predicate = {
            row[0]: row[1]
            for row in self._conn.execute(
                "SELECT predicate, COUNT(*) FROM triples GROUP BY predicate ORDER BY COUNT(*) DESC LIMIT 20"
            ).fetchall()
        }
        return {
            "entities": entity_count,
            "triples": triple_count,
            "by_layer": by_layer,
            "by_predicate": by_predicate,
        }

    def purge_layer(self, layer: str) -> int:
        """Delete all triples in a layer, then any entities left orphaned.

        Lets a caller re-derive an inference/LLM layer without touching source
        facts. Returns the number of triples deleted.
        """
        with self._conn:
            deleted = self._conn.execute(
                "DELETE FROM triples WHERE layer = ?", (layer,)
            ).rowcount
            self._conn.execute(
                """DELETE FROM entities
                   WHERE id NOT IN (SELECT subject FROM triples)
                     AND id NOT IN (SELECT object FROM triples)"""
            )
        return deleted

    def neighbourhood(
        self, anchor: str, max_depth: int = 2, layer: str | None = None
    ) -> list[dict[str, Any]]:
        """Bounded k-hop traversal from ``anchor`` over subject<->object edges.

        Walks up to ``max_depth`` hops via a recursive CTE, optionally confined
        to a single layer. Returns the distinct triple rows touched, as dicts.
        """
        layer_filter = "AND t.layer = :layer" if layer else ""
        sql = f"""
        WITH RECURSIVE reach(node, depth) AS (
            SELECT :anchor, 0
            UNION
            SELECT CASE WHEN t.subject = r.node THEN t.object ELSE t.subject END,
                   r.depth + 1
            FROM reach r
            JOIN triples t ON (t.subject = r.node OR t.object = r.node) {layer_filter}
            WHERE r.depth < :max_depth
        )
        SELECT DISTINCT t.*
        FROM triples t
        JOIN reach r ON (t.subject = r.node OR t.object = r.node)
        {("WHERE t.layer = :layer") if layer else ""}
        """
        params = {"anchor": anchor, "max_depth": max_depth}
        if layer:
            params["layer"] = layer
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
