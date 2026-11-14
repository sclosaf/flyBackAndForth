"""SQLite file inside the project: creation, permissions, schema migrations.

One short-lived connection per operation: LangGraph runs searches in parallel
threads, and sqlite3 connections must not be shared between threads. WAL mode
lets a reader and a writer work at the same time (e.g. a manual run while the
daily price watch is running).
"""
from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

MIGRATIONS: list[str] = [
    """
    CREATE TABLE runs (
        id               INTEGER PRIMARY KEY,
        started_at       TEXT    NOT NULL,             -- ISO 8601, UTC
        finished_at      TEXT,
        provider         TEXT    NOT NULL,
        user_input       TEXT,
        request_json     TEXT,                         -- TripRequest after parsing
        status           TEXT    NOT NULL DEFAULT 'running'
                         CHECK (status IN ('running', 'ok', 'no_request', 'error')),
        network_searches INTEGER NOT NULL DEFAULT 0,   -- calls that used API quota
        errors_json      TEXT    NOT NULL DEFAULT '[]'
    );

    -- One row per search the agent made, whatever its outcome:
    --   fresh  = fetched from the provider now; its offers are stored
    --   cached = served from an earlier fresh search (cached_from)
    --   error  = provider failed or budget exhausted
    CREATE TABLE searches (
        id          INTEGER PRIMARY KEY,
        run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        provider    TEXT    NOT NULL,
        query_key   TEXT    NOT NULL,                  -- ARN-VCE-2026-12-19
        origin      TEXT    NOT NULL,
        destination TEXT    NOT NULL,
        day         TEXT    NOT NULL,
        searched_at TEXT    NOT NULL,
        status      TEXT    NOT NULL CHECK (status IN ('fresh', 'cached', 'error')),
        cached_from INTEGER REFERENCES searches(id) ON DELETE SET NULL,
        error       TEXT
    );
    CREATE INDEX idx_searches_cache
        ON searches (provider, query_key, status, searched_at);

    -- Every itinerary seen in a fresh search: this is the price history.
    -- The same flight seen on different days shares the same fingerprint.
    CREATE TABLE offers (
        id            INTEGER PRIMARY KEY,
        search_id     INTEGER NOT NULL REFERENCES searches(id) ON DELETE CASCADE,
        fingerprint   TEXT    NOT NULL,
        price_eur     REAL    NOT NULL,
        origin        TEXT    NOT NULL,
        destination   TEXT    NOT NULL,
        departure     TEXT    NOT NULL,
        stops         INTEGER NOT NULL,
        airlines      TEXT    NOT NULL,
        total_minutes INTEGER NOT NULL,
        option_json   TEXT    NOT NULL                 -- full FlightOption, re-validated on read
    );
    CREATE INDEX idx_offers_search ON offers (search_id);
    CREATE INDEX idx_offers_fingerprint ON offers (fingerprint);
    CREATE INDEX idx_offers_route ON offers (origin, destination, departure);
    """,
]

SCHEMA_VERSION = len(MIGRATIONS)


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    """Open, run one transaction (commit on success, rollback on error), close."""
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db(path: Path) -> int:
    """Create the database file if needed and bring the schema up to date.

    The directory is created with mode 0700 and the file with mode 0600: the
    database holds your travel plans, so other local users cannot read it.
    Returns the schema version.
    """
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not path.exists():
        os.close(os.open(path, os.O_CREAT | os.O_WRONLY, 0o600))

    with connect(path) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current > SCHEMA_VERSION:
        raise RuntimeError(
            f"{path} has schema v{current}, this code knows up to v{SCHEMA_VERSION}: "
            "update the code, do not downgrade the database"
        )
    for version in range(current + 1, SCHEMA_VERSION + 1):
        with connect(path) as conn:
            conn.executescript(
                "BEGIN;\n" + MIGRATIONS[version - 1] + f"\nPRAGMA user_version = {version};\nCOMMIT;"
            )
    return SCHEMA_VERSION

