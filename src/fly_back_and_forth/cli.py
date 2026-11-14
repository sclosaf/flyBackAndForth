"""Command line entry point.
"""
from __future__ import annotations

import argparse
import logging
from collections import Counter
from datetime import date, timedelta

from fly_back_and_forth.app import run_agent
from fly_back_and_forth.config import PROJECT_ROOT, Settings, load_search_config
from fly_back_and_forth.models import Optimize, RankedOption, TripRequest
from fly_back_and_forth.storage import FlightStore
from fly_back_and_forth.tools.budget import SearchBudget
from fly_back_and_forth.tools.flights import make_provider

def format_option(i: int, o: RankedOption) -> str:
    f = o.flight
    route = " -> ".join([f.segments[0].origin] + [s.destination for s in f.segments])
    airlines = ", ".join(sorted({s.airline for s in f.segments}))
    stops = "direct" if f.stops == 0 else f"{f.stops} stop"

    h, m = divmod(o.door_to_door_minutes, 60)
    extra = f" + access {o.access_cost_eur:.2f}" if o.access_cost_eur else ""

    return (f"{i}. EUR {o.total_cost_eur:>7.2f}  {f.departure:%a %d %b %H:%M}  {route:<17} " f"{stops:<7} {airlines:<20} door-to-door {h}h{m:02d}  " f"(flight {f.price_eur:.2f}{extra})")

def print_stats(store: FlightStore) -> None:
    s = store.stats()
    served = s["fresh"] + s["cached"]
    hit_rate = f"{100 * s['cached'] / served:.0f}%" if served else "-"

    print(f"Database: {s['path']}  ({s['size_kb']} KB, schema v{s['schema_version']})")
    print(f"Runs: {s['runs']}  (first {s['first_run'] or '-'}, last {s['last_run'] or '-'})")
    print(f"Searches: {s['searches']}  fresh {s['fresh']}, cached {s['cached']}, " f"errors {s['errors']}  (cache hit rate {hit_rate})")
    print(f"API calls that used quota: {s['api_calls']}   Offers stored: {s['offers']}")

def print_history(store: FlightStore, route: str) -> None:
    origin, _, destination = route.upper().partition("-")
    rows = store.price_history(origin, destination)

    if not rows:
        print(f"No history for {origin}-{destination} yet.")
        return

    print(f"{'day':<12}{'cheapest':>10}{'seen':>6}   first seen -> last seen")
    for r in rows:
        print(f"{r['day']:<12}{r['cheapest_eur']:>10.2f}{r['observations']:>6}   " f"{r['first_seen'][:16]} -> {r['last_seen'][:16]}")

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="flyBackAndForth")
    p.add_argument("request", nargs="?", help="what you want, in plain language")
    p.add_argument("-o", "--origin", help="IATA codes, comma separated (airports or STO, LON...)")
    p.add_argument("-d", "--destination", help="IATA codes, comma separated")
    p.add_argument("--from", dest="earliest", type=date.fromisoformat, help="first departure day")
    p.add_argument("--to", dest="latest", type=date.fromisoformat, help="last departure day (default: --from + default_window_days)")
    p.add_argument("--max", dest="max_price", type=float, help="max total price in EUR")
    p.add_argument("--max-stops", type=int, help="0 = direct flights only")
    p.add_argument("--optimize", choices=[o.value for o in Optimize], default="price")
    p.add_argument("--refresh", action="store_true", help="ignore cached searches")
    p.add_argument("--stats", action="store_true", help="show database statistics")
    p.add_argument("--history", metavar="ROUTE", help="price history, e.g. ARN-VCE")
    p.add_argument("--prune", metavar="DAYS", type=int, help="delete runs older than DAYS")
    p.add_argument("-v", "--verbose", action="store_true")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    settings = Settings()
    store = FlightStore(settings.db_path)

    if args.stats:
        return print_stats(store)

    if args.history:
        return print_history(store, args.history)

    if args.prune is not None:
        n = store.prune(timedelta(days=args.prune))
        return print(f"Deleted {n} runs older than {args.prune} days.")

    cfg = load_search_config()
    budget = SearchBudget(settings.fly_back_and_forth_max_searches)
    provider = make_provider(settings, budget)

    kwargs: dict = {}
    if args.earliest or args.origin or args.destination:
        if not (args.earliest and args.origin and args.destination):
            p.error("with flags, give all of --origin, --destination and --from")

        kwargs["request"] = TripRequest(origins=args.origin.split(","), destinations=args.destination.split(","), earliest_departure=args.earliest, latest_departure=args.latest, max_price_eur=args.max_price, max_stops=args.max_stops, optimize=args.optimize)

    elif args.request:
        from langchain.chat_models import init_chat_model
        kawrgs["user_input"] = args.request
        kwargs["llm"] = init_chat_model(settings.fly_back_and_forth_model, temperature=0)
    else:
        p.error("give a request in plain language, or --origin, --destination and --form")

    out = run_agent(provider=provider, store=store, cfg=cfg, budget=budget, cache_ttl=timedelta(hours=settings.fly_back_and_forth_cache_ttl_hours), today=date.today(), refresh=args.refresh, **kwargs)

    if out.get("request"):
        r = out["request"]
        start, end = r.window(cfg.default_window_days)
        filters = [f"optimize: {r.optimize.value}", f"baggage: {r.baggage.value}"]

        if r.max_price_eur:
            filters.append(f"max EUR {r.max_price_eur:.0f}")
        if r.max_stops is not None:
            filters.append("directly only" if r.max_stops == 0 else f"max {r.max_stops} stops")

        print(f"{','.join(r.origins)} -> {','.join(r.destinations)}, {start} .. {end}  " f"({', '.join(filters)})  [provider: {provider.name}]")

        log = Counter(out.get("search_log", []))

        print(f"{sum(log.values())} searches: {log['cached']} from cache, " f"{log['fresh']} from the provider, {log['error']} failed  " f"[run #{out['run_id']}, db {settings.db_path.relative_to(PROJECT_ROOT)}]\n")

    for i, option in enumerate(out.get("ranked", []), 1):
        print(format_option(i, option))

    if out.get("request") and not out.get("ranked"):
        print("No flights match the request.")

    for err in out.get("errors", []):
        print(f"! {err}")

if __name__ == "__main__":
    main()
