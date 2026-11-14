"""Wires the nodes into a LangGraph StateGraph.

    START -> parse_request -> plan_searches -> search (x N, parallel) -> rank -> END
                                                 |
                                         cache / FlightStore (SQLite)
"""
from __future__ import annotations

from datetime import timedelta

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from fly_back_and_forth.config import SearchConfig
from fly_back_and_forth.graph import nodes
from fly_back_and_forth.graph.state import AgentState
from fly_back_and_forth.storage import FlightStore
from fly_back_and_forth.tools.flights import FlightProvider


def build_graph(llm: BaseChatModel | None, provider: FlightProvider, cfg: SearchConfig, max_searches: int, store: FlightStore, cache_ttl: timedelta, refresh: bool = False):
    g = StateGraph(AgentState)

    if llm is not None:
        g.add_node("parse_request", nodes.make_parse_request(llm, cfg))
    else:
        g.add_node("parse_request", lambda state: {})

    g.add_node("plan_searches", nodes.make_plan_searches(cfg, max_searches))
    g.add_node("search", nodes.make_search(provider, store, cache_ttl, refresh))
    g.add_node("rank", nodes.make_rank(cfg))

    g.add_edge(START, "parse_request")
    g.add_conditional_edges("parse_request", nodes.route_after_parse, {"plan_searches": "plan_searches", "__end__": END})
    g.add_conditional_edges("plan_searches", nodes.fan_out, ["search"])
    g.add_edge("search", "rank")
    g.add_edge("rank", END)
    return g.compile()

