"""Database, cache across runs, and confinement of data to the project folder."""
import os
import shutil
import sqlite3
import stat
import uuid
from datetime import date, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from fly_back_and_forth import cli
from fly_back_and_forth.app import run_agent
from fly_back_and_forth.config import PROJECT_ROOT, SearchConfig, Settings
from fly_back_and_forth.models import TripRequest
from fly_back_and_forth.storage import FlightStore
from fly_back_and_forth.storage.db import SCHEMA_VERSION
from fly_back_and_forth.tools.budget import SearchBudget
from fly_back_and_forth.tools.flights import FakeProvider

CFG = SearchConfig()
REQUEST = TripRequest(origins=["ARN"], destinations=["VCE", "TRS"],
                      earliest_departure=date(2026, 12, 19),
                      latest_departure=date(2026, 12, 20))   # 2 routes x 2 days = 4 searches
TTL = timedelta(hours=6)


def agent(store, refresh=False, ttl=TTL, provider_cls=FakeProvider, max_searches=10):
    budget = SearchBudget(max_searches)
    out = run_agent(provider=provider_cls(budget), store=store, cfg=CFG, budget=budget,
                    cache_ttl=ttl, today=date(2026, 11, 14), request=REQUEST, refresh=refresh)
    return out, budget


def statuses(store, run_id):
    with sqlite3.connect(store.path) as conn:
        return sorted(r[0] for r in conn.execute(
            "SELECT status FROM searches WHERE run_id = ?", (run_id,)))


# ------------------------------------------------------------ database file
def test_database_is_created_private_and_versioned(store):
    assert store.path.exists()
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.path.parent.stat().st_mode) == 0o700
    assert store.schema_version == SCHEMA_VERSION


def test_reopening_keeps_data_and_does_not_migrate_twice(store, clock):
    agent(store)
    again = FlightStore(store.path, clock=clock)
    assert again.stats()["runs"] == 1


def test_newer_schema_is_refused(store, clock):
    with sqlite3.connect(store.path) as conn:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    with pytest.raises(RuntimeError, match="do not downgrade"):
        FlightStore(store.path, clock=clock)


# ------------------------------------------------------- cache across runs
def test_second_run_is_served_from_cache_without_using_quota(store):
    first, b1 = agent(store)
    second, b2 = agent(store)

    assert b1.used == 4 and b2.used == 0
    assert statuses(store, first["run_id"]) == ["fresh"] * 4
    assert statuses(store, second["run_id"]) == ["cached"] * 4
    assert [o.total_cost_eur for o in first["ranked"]] == \
           [o.total_cost_eur for o in second["ranked"]]


def test_cache_expires_after_ttl(store, clock):
    agent(store)
    clock.advance(hours=5, minutes=59)
    assert agent(store)[1].used == 0            # still valid
    clock.advance(minutes=2)
    out, budget = agent(store)                  # 6h01m after the fresh searches
    assert budget.used == 4
    assert statuses(store, out["run_id"]) == ["fresh"] * 4


def test_refresh_ignores_cache(store):
    agent(store)
    out, budget = agent(store, refresh=True)
    assert budget.used == 4
    assert statuses(store, out["run_id"]) == ["fresh"] * 4


def test_ttl_zero_disables_cache(store):
    agent(store, ttl=timedelta(0))
    assert agent(store, ttl=timedelta(0))[1].used == 4


def test_cache_is_per_provider(store):
    class OtherProvider(FakeProvider):
        name = "other"

    agent(store)
    assert agent(store, provider_cls=OtherProvider)[1].used == 4


def test_corrupted_cache_row_is_a_miss_not_a_crash(store):
    agent(store)
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE offers SET option_json = json_set(option_json, '$.price_eur', -1)")
    out, budget = agent(store)
    assert budget.used == 4                     # fetched again instead of using the bad row
    assert all(o.flight.price_eur > 0 for o in out["ranked"])


# ----------------------------------------------------------- run recording
def test_failures_are_recorded_and_run_continues(store):
    class FlakyProvider(FakeProvider):
        def search(self, query):
            if query.destination == "TRS":
                raise TimeoutError("provider timeout")
            return super().search(query)

    out, _ = agent(store, provider_cls=FlakyProvider)
    assert statuses(store, out["run_id"]) == ["error", "error", "fresh", "fresh"]
    assert out["ranked"]
    with sqlite3.connect(store.path) as conn:
        status, errors = conn.execute("SELECT status, errors_json FROM runs").fetchone()
    assert status == "ok" and "search failed" in errors
    with sqlite3.connect(store.path) as conn:   # details stay in the db, not in the output
        details = [r[0] for r in conn.execute("SELECT error FROM searches WHERE error IS NOT NULL")]
    assert "provider timeout" not in errors
    assert all("provider timeout" in d for d in details)


def test_search_limit_is_recorded_in_the_run(store):
    out, budget = agent(store, max_searches=2)
    assert budget.used == 2
    assert statuses(store, out["run_id"]) == ["fresh", "fresh"]
    with sqlite3.connect(store.path) as conn:
        errors = conn.execute("SELECT errors_json FROM runs").fetchone()[0]
    assert "limit is 2" in errors


def test_crashed_run_is_closed_as_error(store):
    class Boom:
        def with_structured_output(self, schema):
            raise RuntimeError("LLM down")

    budget = SearchBudget(10)
    with pytest.raises(RuntimeError):
        run_agent(provider=FakeProvider(budget), store=store, cfg=CFG, budget=budget,
                  cache_ttl=TTL, today=date(2026, 11, 14), llm=Boom(), user_input="casa")
    with sqlite3.connect(store.path) as conn:
        assert conn.execute("SELECT status FROM runs").fetchone()[0] == "error"


# --------------------------------------------------------------- history
def test_price_history_accumulates_across_runs(store, clock):
    agent(store)
    clock.advance(days=1)
    agent(store)
    rows = store.price_history("ARN", "VCE")
    assert [r["day"] for r in rows] == ["2026-12-19", "2026-12-20"]
    assert all(r["observations"] == 2 for r in rows)


def test_same_itinerary_has_a_price_trend(store, clock):
    out, _ = agent(store)
    clock.advance(days=1)
    agent(store)
    from fly_back_and_forth.storage.store import fingerprint
    trend = store.offer_price_trend(fingerprint(out["results"][0]))
    assert len(trend) == 2


def test_prune_deletes_old_runs_and_their_offers(store, clock):
    agent(store)
    clock.advance(days=100)
    agent(store)
    assert store.prune(timedelta(days=90)) == 1
    s = store.stats()
    assert s["runs"] == 1 and s["offers"] > 0


# -------------------------------------------- confinement to the project
@pytest.mark.parametrize("bad", ["/tmp/fly", "../outside", "var/../../outside"])
def test_data_dir_outside_project_is_rejected(bad, monkeypatch):
    monkeypatch.setenv("FLY_BACK_AND_FORTH_DATA_DIR", bad)
    with pytest.raises(ValidationError, match="outside the project"):
        Settings()


def test_symlink_escaping_the_project_is_rejected(tmp_path, monkeypatch):
    link = PROJECT_ROOT / f".test-link-{uuid.uuid4().hex[:8]}"
    link.symlink_to(tmp_path, target_is_directory=True)
    try:
        monkeypatch.setenv("FLY_BACK_AND_FORTH_DATA_DIR", link.name)
        with pytest.raises(ValidationError, match="outside the project"):
            Settings()
    finally:
        link.unlink()


def test_default_database_is_inside_the_project(monkeypatch):
    monkeypatch.delenv("FLY_BACK_AND_FORTH_DATA_DIR", raising=False)
    assert Settings().db_path == PROJECT_ROOT / "var" / "flyBackAndForth.sqlite"


def test_cli_writes_nothing_outside_the_project(tmp_path, monkeypatch, capsys):
    """Run the real CLI with a fake HOME: the HOME must stay empty."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    data_dir = Path("var") / f"test-{uuid.uuid4().hex[:8]}"
    for var in ("HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(var, str(fake_home))
    monkeypatch.setenv("FLY_BACK_AND_FORTH_DATA_DIR", str(data_dir))
    monkeypatch.setenv("FLY_BACK_AND_FORTH_PROVIDER", "fake")
    try:
        args = ["-o", "ARN", "-d", "VCE", "--from", "2026-12-19", "--to", "2026-12-19"]
        cli.main(args)
        cli.main(args)
        output = capsys.readouterr().out
        assert "1 from cache" in output
        assert (PROJECT_ROOT / data_dir / "flyBackAndForth.sqlite").exists()
        assert list(fake_home.rglob("*")) == []
    finally:
        shutil.rmtree(PROJECT_ROOT / data_dir, ignore_errors=True)
        if (PROJECT_ROOT / "var").exists() and not os.listdir(PROJECT_ROOT / "var"):
            (PROJECT_ROOT / "var").rmdir()

