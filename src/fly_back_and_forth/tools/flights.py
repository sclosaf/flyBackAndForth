"""Flight data providers.

- FakeProvider: deterministic offline data, for development and tests (no quota used).
- SerpApiProvider: real Google Flights data via SerpApi (free plan: 250 searches/month).

Every provider returns validated FlightOption objects. Raw API payloads never
leave this module. Providers only fetch: caching and storage are done by the
search node with FlightStore (fly_back_and_forth.storage), the same way for every provider.
Every call that would use quota goes through SearchBudget first.
"""
from __future__ import annotations

import hashlib
import logging
import random
from datetime import datetime, timedelta
from typing import ClassVar, Protocol

import httpx
from pydantic import ValidationError

from fly_back_and_forth.config import Settings
from fly_back_and_forth.models import FlightOption, SearchQuery, Segment
from fly_back_and_forth.tools.budget import SearchBudget

log = logging.getLogger(__name__)

SERPAPI_URL = "https://serpapi.com/search"
MAX_RESULTS_PER_QUERY = 10

class FlightProvider(Protocol):
    name: str

    def search(self, query: SearchQuery) -> list[FlightOption]: ...

class FakeProvider:
    """Plausible, repeatable results (same query -> same flights). No network."""

    name = "fake"

    def __init__(self, budget: SearchBudget | None = None) -> None:
        self.budget = budget

    AIRLINES: ClassVar[list[str]] = ["SAS", "Norwegian", "Ryanair", "Lufthansa", "Austrian"]

    def search(self, query: SearchQuery) -> list[FlightOption]:
        if self.budget is not None:
            self.budget.spend()
        seed = int(hashlib.sha256(query.key.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        options = []
        for _ in range(rng.randint(3, 6)):
            dep = datetime.combine(query.day, datetime.min.time()) + timedelta(
                hours=rng.randint(6, 21), minutes=rng.choice([0, 15, 30, 45])
            )
            direct = rng.random() < 0.4
            airline = rng.choice(self.AIRLINES)
            if direct:
                arr = dep + timedelta(minutes=rng.randint(150, 175))
                segs = [Segment(origin=query.origin, destination=query.destination,
                                departure=dep, arrival=arr, airline=airline)]
            else:
                hub = rng.choice(["CPH", "FRA", "MUC", "VIE"])
                mid_arr = dep + timedelta(minutes=rng.randint(70, 130))
                mid_dep = mid_arr + timedelta(minutes=rng.randint(45, 240))
                arr = mid_dep + timedelta(minutes=rng.randint(70, 110))
                segs = [
                    Segment(origin=query.origin, destination=hub, departure=dep,
                            arrival=mid_arr, airline=airline),
                    Segment(origin=hub, destination=query.destination, departure=mid_dep,
                            arrival=arr, airline=airline),
                ]
            options.append(FlightOption(
                query_key=query.key,
                price_eur=round(rng.uniform(45, 260) + (40 if direct else 0), 2),
                segments=segs,
                total_minutes=int((arr - dep).total_seconds() // 60),
                source=self.name,
            ))
        return options


def _short(value: object, max_len: int) -> str:
    """Single line, trimmed, at most max_len characters."""
    return " ".join(str(value).split())[:max_len]


def parse_serpapi(payload: dict, query: SearchQuery) -> list[FlightOption]:
    """Turn a SerpApi Google Flights response into validated options.

    Anything that does not validate is dropped and logged, never passed on.
    """
    raw = (payload.get("best_flights") or []) + (payload.get("other_flights") or [])
    options: list[FlightOption] = []
    for item in raw[:MAX_RESULTS_PER_QUERY]:
        try:
            segs = [
                Segment(
                    origin=f["departure_airport"]["id"],
                    destination=f["arrival_airport"]["id"],
                    departure=datetime.strptime(f["departure_airport"]["time"], "%Y-%m-%d %H:%M"),
                    arrival=datetime.strptime(f["arrival_airport"]["time"], "%Y-%m-%d %H:%M"),
                    airline=_short(f.get("airline", "?"), 60),
                    flight_number=_short(f.get("flight_number", ""), 12),
                )
                for f in item["flights"]
            ]
            options.append(FlightOption(
                query_key=query.key,
                price_eur=float(item["price"]),
                segments=segs,
                total_minutes=int(item["total_duration"]),
                source="serpapi",
            ))
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            log.warning("dropping malformed result for %s: %s", query.key, exc)
    return options

class SerpApiProvider:
    name = "serpapi"

    def __init__(self, settings: Settings, budget: SearchBudget) -> None:
        if settings.serpapi_api_key is None:
            raise RuntimeError("SERPAPI_API_KEY is not set (see .env.example)")
        self._key = settings.serpapi_api_key
        self.budget = budget
        self.http = httpx.Client(timeout=30)

    def search(self, query: SearchQuery) -> list[FlightOption]:
        self.budget.spend()
        params = {
            "engine": "google_flights",
            "departure_id": query.origin,
            "arrival_id": query.destination,
            "outbound_date": query.day.isoformat(),
            "type": "2",
            "currency": "EUR",
            "hl": "en",
            "api_key": self._key.get_secret_value(),
        }
        resp = self.http.get(SERPAPI_URL, params=params)
        resp.raise_for_status()
        return parse_serpapi(resp.json(), query)

def make_provider(settings: Settings, budget: SearchBudget) -> FlightProvider:
    if settings.fly_back_and_forth_provider == "serpapi":
        return SerpApiProvider(settings, budget)
    if settings.fly_back_and_forth_provider == "fake":
        return FakeProvider(budget)
    raise ValueError(f"unknown provider {settings.fly_back_and_forth_provider!r}")

