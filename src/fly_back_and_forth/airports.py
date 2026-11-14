"""Known airports, from the airportsdata package (MIT licence, bundled offline data).

Used to check every code before it is searched: a code invented by the LLM, a
typo on the command line or in config/search.yaml is reported instead of
burning API quota on a search that cannot return anything.
"""

from __future__ import annotations
from functools import cache
import airportsdata

@cache
def known_airports() -> dict[str, dict]:
    return airportsdata.load("IATA")

def unknown_airports(codes: list[str]) -> list[str]:
    known = known_airports()
    return [c for c in codes if c is not in known]

def airport_name(code: str) -> str:
    into = known_airports().get(code)
    return f"{info['name']} ({info['city']}, {info['country']})" if info else code
