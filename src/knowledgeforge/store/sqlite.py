"""SQLite graph store — the default GraphStore implementation.

Schema keeps four layers distinct: source_facts / normalised_triples /
inferred_relations / llm_hypotheses. Query by layer or across all layers.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from ..contracts import Triple


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
    layer               TEXT NOT NULL DEFAULT 'source_facts'
);

CREATE INDEX IF NOT EXISTS idx_t_subject   ON triples(subject);
CREATE INDEX IF NOT EXISTS idx_t_object    ON triples(object);
CREATE INDEX IF NOT EXISTS idx_t_predicate ON triples(predicate);
CREATE INDEX IF NOT EXISTS idx_t_layer     ON triples(layer);
CREATE INDEX IF NOT EXISTS idx_t_source    ON triples(source_doc);
"""


class SQLiteGraphStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.commit()

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
                self._conn.execute(
                    "INSERT OR IGNORE INTO entities(id, kind, created_at) VALUES (?,?,?)",
                    (entity_id, kind, triple.timestamp),
                )
            self._conn.execute(
                """INSERT INTO triples(
                    triple_id, subject, predicate, object,
                    source_kind, target_kind, evidence, confidence,
                    source_doc, extraction_method, timestamp, adapter, layer
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    tid,
                    triple.subject, triple.predicate, triple.object,
                    triple.source_kind, triple.target_kind,
                    triple.evidence, triple.confidence,
                    triple.source_doc, triple.extraction_method,
                    triple.timestamp, triple.adapter, triple.layer,
                ),
            )
        return True

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
    ) -> list[dict]:
        clauses: list[str] = []
        params: list = []
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

    def provenance(self, subject: str) -> list[dict]:
        rows = self._conn.execute(
            """SELECT t.*, e.kind as subject_kind
               FROM triples t JOIN entities e ON t.subject = e.id
               WHERE t.subject = ? OR t.object = ?
               ORDER BY t.timestamp""",
            (subject, subject),
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
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

    def close(self) -> None:
        self._conn.close()
