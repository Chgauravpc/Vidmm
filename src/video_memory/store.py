"""
The interval store: SQLite, no server, no vector database.

Why SQLite and not Qdrant/Redis/Neo4j. The queries this system actually needs
are interval-overlap queries over at most a few thousand assertions per video.
That is a B-tree and a range scan. A vector DB would add a service, a container,
and a network hop to a problem that fits in a file — and would make the "clone
the repo and run it" reviewer story impossible.

Open-ended intervals are stored as +inf, which SQLite round-trips exactly and
compares correctly, so `end > :t` works without a sentinel encoding.

The bi-temporal second axis (ingest_time) is stored but not yet indexed; queries
today are all in media time. Keeping the column from the start means "what did
the system believe at ingest time X?" stays answerable later.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Iterable, Sequence

from .types import Assertion, Closure, Evidence, Interval, Seconds

SCHEMA = """
CREATE TABLE IF NOT EXISTS assertions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    subject     TEXT NOT NULL,
    relation    TEXT NOT NULL,
    object      TEXT NOT NULL,
    start       REAL NOT NULL,
    end         REAL NOT NULL,
    closure     TEXT NOT NULL,
    confidence  REAL NOT NULL,
    source      TEXT,
    raw_score   REAL,
    frame_ids   TEXT,
    ingest_time TEXT
);
CREATE INDEX IF NOT EXISTS idx_start    ON assertions(start);
CREATE INDEX IF NOT EXISTS idx_end      ON assertions(end);
CREATE INDEX IF NOT EXISTS idx_relation ON assertions(relation);
CREATE INDEX IF NOT EXISTS idx_object   ON assertions(object);
"""


class IntervalStore:
    def __init__(self, path: str = ":memory:") -> None:
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ---------------- write ----------------
    def add_all(self, assertions: Iterable[Assertion]) -> int:
        rows = [
            (
                a.subject, a.relation, a.object,
                a.interval.start, a.interval.end, a.interval.closure.value,
                a.confidence, a.evidence.source, a.evidence.raw_score,
                json.dumps(list(a.evidence.frame_ids)), a.ingest_time,
            )
            for a in assertions
        ]
        self.conn.executemany(
            "INSERT INTO assertions "
            "(subject, relation, object, start, end, closure, confidence, "
            " source, raw_score, frame_ids, ingest_time) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()
        return len(rows)

    # ---------------- read ----------------
    def at(self, t: Seconds, relation: str | None = None) -> list[Assertion]:
        """Everything that held at instant `t`. The core temporal query."""
        sql = "SELECT * FROM assertions WHERE start <= ? AND end > ?"
        params: list = [t, t]
        if relation:
            sql += " AND relation = ?"
            params.append(relation)
        sql += " ORDER BY confidence DESC"
        return [_row_to_assertion(r) for r in self.conn.execute(sql, params)]

    def during(
        self, start: Seconds, end: Seconds, relation: str | None = None
    ) -> list[Assertion]:
        """Everything overlapping the window [start, end)."""
        sql = "SELECT * FROM assertions WHERE start < ? AND end > ?"
        params: list = [end, start]
        if relation:
            sql += " AND relation = ?"
            params.append(relation)
        sql += " ORDER BY start ASC"
        return [_row_to_assertion(r) for r in self.conn.execute(sql, params)]

    def timeline(self, relation: str, subject: str | None = None) -> list[Assertion]:
        """The full ordered history of one relation — 'when did X happen?'."""
        sql = "SELECT * FROM assertions WHERE relation = ?"
        params: list = [relation]
        if subject:
            sql += " AND subject = ?"
            params.append(subject)
        sql += " ORDER BY start ASC"
        return [_row_to_assertion(r) for r in self.conn.execute(sql, params)]

    def find(self, object_like: str, relation: str | None = None) -> list[Assertion]:
        """'When was there a cup?' — object lookup, substring match."""
        sql = "SELECT * FROM assertions WHERE object LIKE ?"
        params: list = [f"%{object_like}%"]
        if relation:
            sql += " AND relation = ?"
            params.append(relation)
        sql += " ORDER BY start ASC"
        return [_row_to_assertion(r) for r in self.conn.execute(sql, params)]

    def all(self) -> list[Assertion]:
        return [_row_to_assertion(r) for r in self.conn.execute(
            "SELECT * FROM assertions ORDER BY start ASC")]

    def relations(self) -> list[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT DISTINCT relation FROM assertions ORDER BY relation")]

    def objects(self, relation: str | None = None) -> list[str]:
        sql = "SELECT DISTINCT object FROM assertions"
        params: list = []
        if relation:
            sql += " WHERE relation = ?"
            params.append(relation)
        return [r[0] for r in self.conn.execute(sql + " ORDER BY object", params)]

    def span(self) -> tuple[Seconds, Seconds]:
        row = self.conn.execute(
            "SELECT MIN(start), MAX(CASE WHEN end = 9e999 THEN start ELSE end END) "
            "FROM assertions").fetchone()
        return (row[0] or 0.0, row[1] or 0.0)

    def __len__(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM assertions").fetchone()[0]

    def close(self) -> None:
        self.conn.close()


def _row_to_assertion(r: sqlite3.Row) -> Assertion:
    return Assertion(
        subject=r["subject"],
        relation=r["relation"],
        object=r["object"],
        interval=Interval(
            start=r["start"], end=r["end"], closure=Closure(r["closure"])
        ),
        confidence=r["confidence"],
        evidence=Evidence(
            frame_ids=tuple(json.loads(r["frame_ids"] or "[]")),
            source=r["source"] or "",
            raw_score=r["raw_score"],
        ),
        ingest_time=r["ingest_time"],
    )


def store_from_pack(pack_path: str, db_path: str = ":memory:") -> IntervalStore:
    """Load a committed feature pack into a queryable store. No GPU, no torch."""
    from .pack import read_pack  # local import keeps store.py dependency-free

    pack = read_pack(pack_path)
    store = IntervalStore(db_path)
    store.add_all(pack["assertions"])
    return store
