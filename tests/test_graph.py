"""End-to-end graph runs, offline: FakeProvider and a fake LLM. No API keys needed."""
from datetime import date, timedelta

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from pydantic import Field, ValidationError

from fly_back_and_forth.config import AirportAccess, SearchConfig
from fly_back_and_forth.graph import build_graph
from fly_back_and_forth.models import Optimize, TripRequest
from fly_back_and_forth.tools.flights import FakeProvider

CFG = SearchConfig(
    default_window_days=3,
    max_airports_per_side=3,
    metro_areas={"STO": ["ARN", "BMA", "NYO"], "LON": ["LHR", "LGW", "STN", "LTN"]},
    airport_access={"NYO": AirportAccess(cost_eur=25, minutes=80)},
)
TODAY = date(2026, 11, 14)
REQUEST = TripRequest(origins=["ARN", "NYO"], destinations=["VCE", "TRS"],
                      earliest_departure=date(2026, 12, 19),
                      latest_departure=date(2026, 12, 20), max_price_eur=400)
TTL = timedelta(hours=6)


def run(store, request=REQUEST, max_searches=50, cfg=CFG):
    graph = build_graph(None, FakeProvider(), cfg, max_searches, store, TTL)
    return graph.invoke({"run_id": store.start_run("fake"), "today": TODAY, "request": request})


def req(**changes):
    return TripRequest(**(REQUEST.model_dump() | changes))


# ------------------------------------------------------------- planning
def test_parallel_search_covers_every_route_and_day(store):
    out = run(store)
    assert len(out["queries"]) == 2 * 2 * 2
    assert {r.query_key for r in out["results"]} == {q.key for q in out["queries"]}


def test_metro_code_expands_to_its_airports(store):
    out = run(store, req(origins=["STO"], destinations=["VCE"]))
    assert {q.origin for q in out["queries"]} == {"ARN", "BMA", "NYO"}


def test_airports_per_side_are_capped_and_reported(store):
    out = run(store, req(origins=["LON"], destinations=["VCE"]))
    assert [q.origin for q in out["queries"][:3]] == ["LHR", "LGW", "STN"]
    assert any("skipped LTN" in e for e in out["errors"])


def test_unknown_airport_is_rejected_before_any_search(store):
    out = run(store, request=req(origins=["XQZ"]))
    assert out["queries"] == [] and not out.get("results")
    assert any("unknown airport code(s): XQZ" in e for e in out["errors"])


def test_typo_in_metro_area_config_is_caught(store):
    bad_cfg = CFG.model_copy(update={"metro_areas": {"STO": ["ARN", "QQQ"]}})
    out = run(store, request=req(origins=["STO"]), cfg=bad_cfg)
    assert out["queries"] == []
    assert any("QQQ" in e for e in out["errors"])


def test_start_date_only_searches_the_default_window(store):
    out = run(store, req(origins=["ARN"], destinations=["VCE"], latest_departure=None))
    assert sorted(q.day for q in out["queries"]) == [
        date(2026, 12, 19), date(2026, 12, 20), date(2026, 12, 21)]


def test_past_dates_are_skipped(store):
    out = run(store, req(origins=["ARN"], destinations=["VCE"],
                         earliest_departure=TODAY - timedelta(days=2), latest_departure=TODAY))
    assert [q.day for q in out["queries"]] == [TODAY]
    assert any("past dates" in e for e in out["errors"])


def test_search_limit_keeps_earliest_days_complete(store):
    out = run(store, max_searches=5)
    days = [q.day for q in out["queries"]]
    assert days.count(date(2026, 12, 19)) == 4           # all 4 routes on the first day
    assert any("searched up to 2026-12-20" in e for e in out["errors"])


# -------------------------------------------------------------- ranking
def test_price_ranking_includes_airport_access(store):
    ranked = run(store)["ranked"]
    totals = [o.total_cost_eur for o in ranked]
    assert totals == sorted(totals)
    for o in ranked:
        assert o.access_cost_eur == (25 if o.flight.origin == "NYO" else 0)


def test_direct_only(store):
    ranked = run(store, req(max_stops=0))["ranked"]
    assert ranked and all(o.flight.stops == 0 for o in ranked)


def test_optimize_duration(store):
    ranked = run(store, req(optimize=Optimize.DURATION))["ranked"]
    minutes = [o.door_to_door_minutes for o in ranked]
    assert minutes == sorted(minutes)


def test_optimize_value_sits_between_price_and_duration(store):
    """Time worth 0 EUR/h -> same as price; time worth a fortune -> same as duration."""
    def first(optimize, eur_per_hour):
        cfg = CFG.model_copy(update={"value_of_time_eur_per_hour": eur_per_hour, "top_n": 50})
        return run(store, req(optimize=optimize), cfg=cfg)["ranked"][0].flight

    assert first(Optimize.VALUE, 0) == first(Optimize.PRICE, 0)
    assert first(Optimize.VALUE, 1_000_000) == first(Optimize.DURATION, 0)


def test_budget_filter(store):
    assert run(store, req(max_price_eur=1))["ranked"] == []


# ------------------------------------------------------------- request
def test_codes_are_normalised_and_validated():
    r = req(origins=[" arn "], destinations=["vce"])
    assert r.origins == ["ARN"] and r.destinations == ["VCE"]
    with pytest.raises(ValidationError):
        req(origins=["ARLANDA"])
    with pytest.raises(ValidationError):
        req(origins=["ARN", "BMA", "NYO", "VST"])                # max 3 codes per side


def test_same_origin_and_destination_is_rejected():
    with pytest.raises(ValidationError, match="same"):
        req(origins=["VCE"], destinations=["VCE"])


def test_request_rejects_reversed_or_huge_windows():
    with pytest.raises(ValidationError):
        req(earliest_departure=date(2026, 12, 20), latest_departure=date(2026, 12, 18))
    with pytest.raises(ValidationError):
        req(earliest_departure=date(2026, 12, 1), latest_departure=date(2026, 12, 31))


# ------------------------------------------------------ natural language
class _StructuredFake(GenericFakeChatModel):
    """Fake LLM whose structured output returns a fixed TripRequest."""

    prompts: list = Field(default_factory=list)

    def with_structured_output(self, schema, **kwargs):
        from langchain_core.runnables import RunnableLambda

        def answer(prompt):
            self.prompts.append(prompt)
            return REQUEST
        return RunnableLambda(answer)


def test_natural_language_path_with_fake_llm(store):
    llm = _StructuredFake(messages=iter([]))
    graph = build_graph(llm, FakeProvider(), CFG, 50, store, TTL)
    out = graph.invoke({"run_id": store.start_run("fake"), "today": TODAY,
                        "user_input": "da Stoccolma a Venezia il 19 o 20 dicembre, max 400"})
    assert out["request"] == REQUEST
    assert out["ranked"]
    assert "STO" in llm.prompts[0] and "2026-11-14" in llm.prompts[0]

