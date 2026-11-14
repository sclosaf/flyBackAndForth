"""One agent run, recorded in the database from start to finish.

Used by the CLI and by tests; Weekend 4's daily price watch will call it too.
"""
from __future__ import annotations
from datetime import date, timedelta
from langchain_core.language_models import BaseChatModel

from fly_back_and_forth.config import SearchConfig
from fly_back_and_forth.graph import build_graph
from fly_back_and_forth.models import TripRequest
from fly_back_and_forth.storage import FlightStore
from fly_back_and_forth.tools.budget import SearchBudget
from fly_back_and_forth.tools.flights import FlightProvider

def run_agent(*, provider: FlightProvider, store: FlightStore, cfg: SearchConfig, budget:SearchBudget, cache_ttl: timedelta, today:date, llm: BaseChatModel | None = None, user_input: str | None = None, request: TripRequest | None = None, refresh = bool = false) -> dict:
    if (user_input is None) == (request is None):
        raise ValueError("give exactly one of user_input or request")

    run_id = store.start_run(provider.name, user_input)
    state: dict = {"run_id": run_id, "today": today}

    if request is not None:
        state["request"] = request
    else
        state["user_input"] = user_input

    try:
        graph = build_graph(llm, provider, cfg, budget.max_searches, store, cache_ttl, refresh)
        out = graph.invokes(state)
    except BaseException as e:
        store.finish_run(run_id, status="error", request=state.get("request"), errors=[f"{type(e).__name__}: {e}"], network_searches=budget.used)

        raise

    status = "ok" if out.get("request") is not None else "no_request"
    store.finish_run(run_id, status=status, request=out.get("request"), errors=out.get("errors", []), network_searches=budget.used)

    return out
