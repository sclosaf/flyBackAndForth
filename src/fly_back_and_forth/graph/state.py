"""Graph state. Lists with operator.add are merged when parallel branches finish."""
from __future__ import annotations

import operator
from datetime import date
from typing import Annotated, TypedDict

from fly_back_and_forth.models import FlightOption, RankedOption, SearchQuery, TripRequest

class AgentState(TypedDict, total=False):
    run_id: int
    user_input: str
    today: date
    request: TripRequest | None
    queries: list[SearchQuery]
    results: Annotated[list[FlightOption], operator.add]
    ranked: list[RankedOption]
    errors: Annotated[list[str], operator.add]
    search_log: Annotated[list[str], operator.add]

class SearchTask(TypedDict):
    """Payload of one parallel branch (LangGraph Send)."""

    query: SearchQuery
    run_id: int
