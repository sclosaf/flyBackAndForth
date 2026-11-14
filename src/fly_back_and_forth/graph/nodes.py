"""Graph nodes. Each one is a plain function: easy to test without the graph."""
from __future__ import annotations

import logging
from datetime import timedelta
from itertools import product

from langchain_core.language_models import BaseChatModel
from pydantic import ValidationError

from fly_back_and_forth.airports import unknown_airports
from fly_back_and_forth.config import SearchConfig
from fly_back_and_forth.graph.state import AgentState, SearchTask
from fly_back_and_forth.models import Optimize, RankedOption, SearchQuery, TripRequest
from fly_back_and_forth.storage import FlightStore
from fly_back_and_forth.tools.budget import BudgetExceeded
from fly_back_and_forth.tools.flights import FlightProvider

log = logging.getLogger(__name__)

PARSE_PROMPT = """You extract flight search requests into a structured form.
Today is {today}.
- origins, destinations: IATA codes. If the user names a city with several airports
  and no specific airport, use its metropolitan code ({metro_codes}); otherwise the
  code of the city's main airport. At most 3 codes per side.
- Resolve relative dates ("next Friday", "for Christmas") against today.
  If the user gives only a start date ("from 10 March"), leave latest_departure empty.
- max_price_eur, max_stops: only if the user says so ("direct only" = 0 stops).
- optimize: "price" by default, "duration" if they want the fastest flight,
  "value" if they want a balance of price and travel time.
- baggage: "cabin" if not mentioned.
Only extract what the user said; do not invent preferences.

User request:
{user_input}"""

def make_parse_request(llm: BaseChatModel, cfg: SearchConfig):
    structured = llm.with_structured_output(TripRequest)

    def parse_request(state: AgentState) -> dict:
        if state.get("request") is not None:
            return {}
        prompt = PARSE_PROMPT.format(
            today=state["today"].isoformat(),
            metro_codes=", ".join(cfg.metro_areas) or "none configured",
            user_input=state["user_input"],
        )
        try:
            request = structured.invoke(prompt)
        except ValidationError as exc:
            return {"request": None, "errors": [f"could not understand the request: {exc}"]}
        return {"request": request}

    return parse_request

def _expand_side(codes: list[str], cfg: SearchConfig, side: str) -> tuple[list[str], list[str]]:
    airports = cfg.expand(codes)
    if len(airports) > cfg.max_airports_per_side:
        dropped = airports[cfg.max_airports_per_side:]
        airports = airports[:cfg.max_airports_per_side]
        warning = (f"{side}: searching {', '.join(airports)}; skipped {', '.join(dropped)} "
                   f"(max_airports_per_side = {cfg.max_airports_per_side})")
        return airports, [warning]
    return airports, []


def make_plan_searches(cfg: SearchConfig, max_searches: int):
    """Turn the request into concrete searches. Pure code: the LLM has no say here."""

    def plan_searches(state: AgentState) -> dict:
        req: TripRequest = state["request"]
        today = state["today"]
        errors: list[str] = []

        origins, e1 = _expand_side(req.origins, cfg, "origins")
        destinations, e2 = _expand_side(req.destinations, cfg, "destinations")
        errors += e1 + e2

        unknown = unknown_airports(origins + destinations)
        if unknown:
            message = (f"unknown airport code(s): {', '.join(unknown)}. Use real IATA codes "
                       f"or a metro code from config/search.yaml ({', '.join(cfg.metro_areas)})")
            return {"queries": [], "errors": [*errors, message]}

        start, end = req.window(cfg.default_window_days)
        if end < today:
            return {"queries": [], "errors": errors + [f"all dates ({start} .. {end}) are past"]}
        if start < today:
            errors.append(f"skipped past dates {start} .. {today - timedelta(days=1)}")
            start = today
        days = [start + timedelta(days=i) for i in range((end - start).days + 1)]

        queries = [
            SearchQuery(origin=o, destination=d, day=day)
            for day, o, d in product(days, origins, destinations)
            if o != d
        ]
        if len(queries) > max_searches:
            last = queries[max_searches - 1].day
            errors.append(
                f"{len(queries)} searches needed, limit is {max_searches}: "
                f"searched up to {last}. Narrow the dates or airports, or raise the limit."
            )
            queries = queries[:max_searches]
        return {"queries": queries, "errors": errors}

    return plan_searches


def fan_out(state: AgentState):
    """Conditional edge: one parallel 'search' branch per query (LangGraph Send)."""
    from langgraph.types import Send

    return [Send("search", SearchTask(query=q, run_id=state["run_id"]))
            for q in state["queries"]]

def make_search(provider: FlightProvider, store: FlightStore, cache_ttl: timedelta,
                refresh: bool = False):
    """Cache first, provider second; every outcome is recorded in the database.

    - cache hit (fresh search for the same provider + route + day within the TTL):
      reuse its offers, record a 'cached' search, no quota used
    - miss, expired or refresh=True: call the provider, store search + offers
    - failure: record an 'error' search; the other branches keep going
    """

    def search(task: SearchTask) -> dict:
        q, run_id = task["query"], task["run_id"]

        if not refresh and cache_ttl > timedelta(0):
            hit = store.cache_lookup(provider.name, q, cache_ttl)
            if hit is not None:
                store.record_cached(run_id, provider.name, q, hit)
                log.info("cache hit %s (age %s)", q.key, hit.age(store.clock()))
                return {"results": hit.options, "search_log": ["cached"]}

        try:
            options = provider.search(q)
        except BudgetExceeded as exc:
            store.record_error(run_id, provider.name, q, str(exc))
            return {"errors": [f"{q.key}: {exc}"], "search_log": ["error"]}
        except Exception as exc:
            log.exception("search failed for %s", q.key)
            store.record_error(run_id, provider.name, q, f"{type(exc).__name__}: {exc}")
            return {"errors": [f"{q.key}: search failed ({type(exc).__name__})"],
                    "search_log": ["error"]}

        store.record_fresh(run_id, provider.name, q, options)
        return {"results": options, "search_log": ["fresh"]}

    return search

def make_rank(cfg: SearchConfig):
    def score(option: RankedOption, optimize: Optimize) -> tuple:
        hours = option.door_to_door_minutes / 60
        if optimize is Optimize.DURATION:
            return (option.door_to_door_minutes, option.total_cost_eur)
        if optimize is Optimize.VALUE:
            return (option.total_cost_eur + hours * cfg.value_of_time_eur_per_hour,)
        return (option.total_cost_eur, option.door_to_door_minutes)

    def rank(state: AgentState) -> dict:
        req: TripRequest = state["request"]
        ranked = []
        for f in state.get("results", []):
            if req.max_stops is not None and f.stops > req.max_stops:
                continue
            access = [cfg.airport_access.get(code) for code in (f.origin, f.destination)]

            option = RankedOption(flight=f, access_cost_eur=sum(a.cost_eur for a in access if a), door_to_door_minutes=f.total_minutes + sum(a.minutes for a in access if a))
            if req.max_price_eur is None or option.total_cost_eur <= req.max_price_eur:
                ranked.append(option)
        ranked.sort(key=lambda o: score(o, req.optimize))
        return {"ranked": ranked[:cfg.top_n]}

    return rank

def route_after_parse(state: AgentState) -> str:
    return "plan_searches" if state.get("request") is not None else "__end__"
