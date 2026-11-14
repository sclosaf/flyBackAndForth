# flyBackAndForth

A LangGraph agent that finds the **best-value flight on a route**, in a date window
or from a given date onwards. Not just the ticket price: it compares airports of the
same city, adds the cost and time to reach each airport, and can rank by price,
duration or a balance of the two.

```
$ flyBackAndForth "best balance of price and duration from Arlanda to London between 9 and 11 January, max 120 euros"
ARN -> LON, 2027-01-09 .. 2027-01-11  (optimize: value, baggage: cabin, max EUR 120)  [provider: serpapi]
9 searches: 3 from cache, 6 from the provider, 0 failed  [run #12, db var/flyBackAndForth.sqlite]

1. EUR   64.90  Sat 09 Jan 07:05  ARN -> STN        direct  Ryanair              door-to-door 3h55  (flight 49.90 + access 15.00)
2. EUR   92.00  Sun 10 Jan 10:15  ARN -> LHR        direct  SAS                  door-to-door 2h35  (flight 92.00)
3. EUR   71.40  Mon 11 Jan 06:20  ARN -> CPH -> LGW 1 stop  Norwegian            door-to-door 5h10  (flight 71.40)
```

What you can ask:

| You say | The agent searches |
|---|---|
| a route: airports (`ARN`), several (`ARN,BMA`) or a metro code (`STO` = all Stockholm airports) | every origin × destination airport, up to `max_airports_per_side` |
| a window: "between 19 and 22 December" | every day in the window (max 14 days) |
| a start date: "from 19 December" | `default_window_days` days starting that day |
| "direct only", "max 150 euro" | filters on stops and total price |
| "the fastest", "best compromise" | ranking by duration, or price + time valued at `value_of_time_eur_per_hour` |

## Architecture

```
            ┌────────────────┐   structured output (Pydantic)
 user text ─▶ parse_request  │── LLM: OpenAI or local Ollama
            └───────┬────────┘
            ┌───────▼────────┐   code, not the LLM: expands metro codes, checks every
            │ plan_searches  │   airport exists, builds airports × days, caps searches
            └───────┬────────┘
       ┌────────────┼────────────┐  LangGraph Send: parallel fan-out
  ┌────▼───┐   ┌────▼───┐   ┌────▼───┐
  │ search │   │ search │   │ search │   cache hit? reuse : SerpApi (Google Flights)
  └────┬───┘   └────┬───┘   └────┬───┘   every result validated and stored in var/ (SQLite)
       └────────────┼────────────┘
            ┌───────▼────────┐
            │      rank      │   total cost = flight + airport access (+ bags: Weekend 3)
            └───────┬────────┘   sorted by price, duration or value
                    ▼
       Weekend 3: confirm_watch (interrupt, human-in-the-loop) → save_watch
```

| Path | What it is |
|---|---|
| `src/fly_back_and_forth/models.py` | Pydantic contracts: `TripRequest`, `SearchQuery`, `FlightOption`, `RankedOption` |
| `src/fly_back_and_forth/graph/` | LangGraph state, nodes and wiring |
| `src/fly_back_and_forth/airports.py` | Check that every airport code exists (offline `airportsdata` list) |
| `src/fly_back_and_forth/tools/flights.py` | `FakeProvider` (offline) and `SerpApiProvider` (real data) |
| `src/fly_back_and_forth/tools/budget.py` | Hard cap on provider calls per run, to protect the API quota |
| `src/fly_back_and_forth/storage/` | SQLite database: schema and migrations (`db.py`), cache and history (`store.py`) |
| `src/fly_back_and_forth/app.py` | One agent run, recorded in the database from start to finish |
| `config/search.yaml` | Your rules: metro areas, airport access costs, window, ranking |
| `data/knowledge/` | Notes for the RAG step (Weekend 3) |

## Setup

```bash
uv venv && uv pip install -e ".[dev]"      # add ",local" for Ollama
cp .env.example .env                       # fill in the keys
pytest                                     # runs fully offline
```

Run without any API key (fake flights, route and dates as flags, no LLM):

```bash
flyBackAndForth -o STO -d VCE --from 2026-12-19                   # 7 days from the 19th
flyBackAndForth -o ARN -d LON --from 2026-12-19 --to 2026-12-21 --max-stops 0
flyBackAndForth -o STO -d MIL --from 2027-01-10 --optimize value --max 150
```

Database commands:

```bash
flyBackAndForth -o STO -d VCE --from 2026-12-19 --refresh   # ignore the cache, fetch again
flyBackAndForth --stats                           # runs, searches, cache hit rate, quota used
flyBackAndForth --history ARN-VCE                 # cheapest price seen per departure day
flyBackAndForth --prune 90                        # delete runs older than 90 days
```

Run with real data: set `FLY_BACK_AND_FORTH_PROVIDER=serpapi`, `SERPAPI_API_KEY` and an LLM
(`OPENAI_API_KEY`, or `FLY_BACK_AND_FORTH_MODEL=ollama:llama3.1` with Ollama running).
Each airport pair × day is one search: `STO -> VCE` over 7 days is 21 searches, so keep
`FLY_BACK_AND_FORTH_MAX_SEARCHES` in mind (SerpApi free plan: 250 searches/month).
When the plan is cut, the earliest days are searched completely first.

> Amadeus Self-Service, used in many tutorials, was shut down on 17 July 2026.
> This project uses SerpApi's Google Flights engine.

## Data and caching

All data lives in **one SQLite file inside the project**: `var/flyBackAndForth.sqlite`.
It is created on the first run, git-ignored, and readable only by you
(file `0600`, folder `0700`). Nothing is written to your home folder, `~/.cache`
or `/tmp`: `FLY_BACK_AND_FORTH_DATA_DIR` must resolve inside the project, otherwise the app
refuses to start. To reset everything, delete `var/`.

| Table | One row per | Used for |
|---|---|---|
| `runs` | execution of the agent | audit: input, parsed request, status, errors, API calls used |
| `searches` | route + day searched in a run | cache lookups; status `fresh`, `cached` or `error` |
| `offers` | itinerary returned by a fresh search | price history; `fingerprint` links the same flight across runs |

**Between two runs**, cache and history are the same data read in two ways:

- **Cache:** before calling the provider, the search node looks for a `fresh`
  search with the same provider, route and day newer than `FLY_BACK_AND_FORTH_CACHE_TTL_HOURS`
  (default 6 h). If found, it reuses those offers and records a `cached` search
  that points to it: no API call, no quota used. `--refresh` skips this step.
- **History:** every fresh search keeps its offers forever (until `--prune`), so
  the next runs can see how prices move. The cache never deletes history.

Data read back from the file is validated with Pydantic like API data: a row
that was corrupted or edited by hand is treated as a cache miss and fetched again.
Schema changes are migrations in `storage/db.py` (`PRAGMA user_version`):
existing databases are upgraded in place, never recreated.

## Design choices

- **The LLM understands, the code decides.** The model only turns free text into a
  `TripRequest`. Expanding metro codes, picking days, capping searches and ranking are
  plain Python: predictable, testable, and no tokens spent on arithmetic.
- **Validate at the edges.** LLM output, API responses and rows read from the
  database all go through Pydantic models; airport codes are checked against a list
  of real airports. Bad data fails early with a clear message.
- **Quota first.** Searches are planned before they run, capped per run
  (`SearchBudget`), cached in SQLite and reused across runs.
- **Read-only agent.** It searches and ranks; it never books or pays.
- **Everything offline-testable.** `FakeProvider` and a fake LLM run the whole graph
  without keys or network: `pytest` covers planning, parallel search, cache and ranking.
