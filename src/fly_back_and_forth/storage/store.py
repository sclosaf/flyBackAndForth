"""FlightStore: runs, searches and offers in one SQLite file.

Cache and history are the same data, read in two ways:

- Cache: "is there a *fresh* search for this provider + route + day newer than
  the TTL?" If yes, its offers are reused and the run records a `cached` search
  pointing to it. No API call, no quota used.
- History: every fresh search adds its offers permanently, so prices of the
  same flight can be compared across runs (price watch, Weekend 4).

Data read back from the file is re-validated with Pydantic, like API data:
a corrupted row is treated as a cache miss instead of breaking the graph.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import ValidationError

from fly_back_and_forth.models import FlightOption, SearchQuery, TripRequest
from fly_back_and_forth.storage.db import connect, init_db

log = logging.getLogger(__name__)


def utcnow() -> datetime:
    return datetime.now(UTC)


def fingerprint(option: FlightOption) -> str:
    """Stable identity of an itinerary, independent of its price."""
    parts = [
        f"{s.origin}{s.destination}{s.departure.isoformat()}{s.airline}{s.flight_number}"
        for s in option.segments
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class CacheHit:
    search_id: int
    searched_at: datetime
    options: list[FlightOption]

    def age(self, now: datetime) -> timedelta:
        return now - self.searched_at


class FlightStore:
    def __init__(self, path: Path, clock: Callable[[], datetime] = utcnow) -> None:
        self.path = path
        self.clock = clock
        self.schema_version = init_db(path)

    def _now(self) -> str:
        return self.clock().isoformat(timespec="microseconds")

    def start_run(self, provider: str, user_input: str | None = None) -> int:
        with connect(self.path) as conn:
            cur = conn.execute(
                "INSERT INTO runs (started_at, provider, user_input) VALUES (?, ?, ?)",
                (self._now(), provider, user_input),
            )
            return cur.lastrowid

    def finish_run(self, run_id: int, *, status: str, request: TripRequest | None,
                   errors: list[str], network_searches: int) -> None:
        with connect(self.path) as conn:
            conn.execute(
                """UPDATE runs SET finished_at = ?, status = ?, request_json = ?,
                                   errors_json = ?, network_searches = ?
                   WHERE id = ?""",
                (self._now(), status,
                 request.model_dump_json() if request else None,
                 json.dumps(errors), network_searches, run_id),
            )

    def cache_lookup(self, provider: str, query: SearchQuery, ttl: timedelta) -> CacheHit | None:
        oldest_valid = (self.clock() - ttl).isoformat(timespec="microseconds")
        with connect(self.path) as conn:
            row = conn.execute(
                """SELECT id, searched_at FROM searches
                   WHERE provider = ? AND query_key = ? AND status = 'fresh'
                     AND searched_at >= ?
                   ORDER BY searched_at DESC LIMIT 1""",
                (provider, query.key, oldest_valid),
            ).fetchone()
            if row is None:
                return None
            raw = conn.execute(
                "SELECT option_json FROM offers WHERE search_id = ? ORDER BY id", (row["id"],)
            ).fetchall()
        try:
            options = [FlightOption.model_validate_json(r["option_json"]) for r in raw]
        except ValidationError:
            log.warning("cached search %s failed validation: treating as a miss", row["id"])
            return None
        return CacheHit(row["id"], datetime.fromisoformat(row["searched_at"]), options)

    def _insert_search(self, conn, run_id: int, provider: str, query: SearchQuery,
                       status: str, cached_from: int | None = None,
                       error: str | None = None) -> int:
        cur = conn.execute(
            """INSERT INTO searches (run_id, provider, query_key, origin, destination, day,
                                     searched_at, status, cached_from, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, provider, query.key, query.origin, query.destination,
             query.day.isoformat(), self._now(), status, cached_from, error),
        )
        return cur.lastrowid

    def record_fresh(self, run_id: int, provider: str, query: SearchQuery,
                     options: list[FlightOption]) -> int:
        with connect(self.path) as conn:
            search_id = self._insert_search(conn, run_id, provider, query, "fresh")
            conn.executemany(
                """INSERT INTO offers (search_id, fingerprint, price_eur, origin, destination,
                                       departure, stops, airlines, total_minutes, option_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (search_id, fingerprint(o), o.price_eur, o.origin, o.destination,
                     o.departure.isoformat(), o.stops,
                     ", ".join(sorted({s.airline for s in o.segments})),
                     o.total_minutes, o.model_dump_json())
                    for o in options
                ],
            )
            return search_id

    def record_cached(self, run_id: int, provider: str, query: SearchQuery, hit: CacheHit) -> int:
        with connect(self.path) as conn:
            return self._insert_search(conn, run_id, provider, query, "cached",
                                       cached_from=hit.search_id)

    def record_error(self, run_id: int, provider: str, query: SearchQuery, error: str) -> int:
        with connect(self.path) as conn:
            return self._insert_search(conn, run_id, provider, query, "error",
                                       error=error[:500])

    def price_history(self, origin: str, destination: str,
                      provider: str | None = None) -> list[dict]:
        """Cheapest price seen per departure day, and how it changed across observations."""
        sql = """
            SELECT substr(o.departure, 1, 10)   AS day,
                   MIN(o.price_eur)             AS cheapest_eur,
                   COUNT(DISTINCT s.id)         AS observations,
                   MIN(s.searched_at)           AS first_seen,
                   MAX(s.searched_at)           AS last_seen
            FROM offers o JOIN searches s ON s.id = o.search_id
            WHERE o.origin = ? AND o.destination = ?
        """
        params: list = [origin, destination]
        if provider:
            sql += " AND s.provider = ?"
            params.append(provider)
        sql += " GROUP BY day ORDER BY day"
        with connect(self.path) as conn:
            return [dict(r) for r in conn.execute(sql, params)]

    def offer_price_trend(self, fp: str) -> list[tuple[str, float]]:
        """(searched_at, price) for one itinerary over time: the price watch input."""
        with connect(self.path) as conn:
            return [
                (r["searched_at"], r["price_eur"])
                for r in conn.execute(
                    """SELECT s.searched_at, o.price_eur FROM offers o
                       JOIN searches s ON s.id = o.search_id
                       WHERE o.fingerprint = ? ORDER BY s.searched_at""",
                    (fp,),
                )
            ]

    def stats(self) -> dict:
        with connect(self.path) as conn:
            row = conn.execute(
                """SELECT
                     (SELECT COUNT(*) FROM runs)                                 AS runs,
                     (SELECT COUNT(*) FROM searches)                             AS searches,
                     (SELECT COUNT(*) FROM searches WHERE status = 'fresh')      AS fresh,
                     (SELECT COUNT(*) FROM searches WHERE status = 'cached')     AS cached,
                     (SELECT COUNT(*) FROM searches WHERE status = 'error')      AS errors,
                     (SELECT COUNT(*) FROM offers)                               AS offers,
                     (SELECT COALESCE(SUM(network_searches), 0) FROM runs)       AS api_calls,
                     (SELECT MIN(started_at) FROM runs)                          AS first_run,
                     (SELECT MAX(started_at) FROM runs)                          AS last_run"""
            ).fetchone()
        result = dict(row)
        result["path"] = str(self.path)
        result["size_kb"] = round(self.path.stat().st_size / 1024, 1)
        result["schema_version"] = self.schema_version
        return result

    def prune(self, older_than: timedelta) -> int:
        """Delete runs (and their searches and offers) older than the given age."""
        cutoff = (self.clock() - older_than).isoformat(timespec="microseconds")
        with connect(self.path) as conn:
            deleted = conn.execute("DELETE FROM runs WHERE started_at < ?", (cutoff,)).rowcount
        with connect(self.path) as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return deleted

