"""Search budget: a hard cap on provider calls per run.

The free SerpApi plan has 250 searches a month, so a run must never use more
than FLY_BACK_AND_FORTH_MAX_SEARCHES, even if the plan or a loop goes wrong.
Cache hits do not count.
"""
from __future__ import annotations

class BudgetExceeded(RuntimeError):
    pass

class SearchBudget:
    def __init__(self, max_searches: int) -> None:
        self.max_searches = max_searches
        self.used = 0

    def spend(self) -> None:
        if self.used >= self.max_searches:
            raise BudgetExceeded(f"search budget of {self.max_searches} exhausted")
        self.used += 1
