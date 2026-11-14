import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from fly_back_and_forth.models import SearchQuery
from fly_back_and_forth.tools.budget import BudgetExceeded, SearchBudget
from fly_back_and_forth.tools.flights import parse_serpapi

FIXTURE = Path(__file__).parent / "fixtures" / "serpapi_arn_vce.json"
QUERY = SearchQuery(origin="ARN", destination="VCE", day=date(2026, 12, 19))


def test_parse_serpapi_keeps_valid_results_only():
    options = parse_serpapi(json.loads(FIXTURE.read_text()), QUERY)
    # 4 raw results: 1 valid connection, 1 valid with a messy airline name,
    # 1 arriving before departing (dropped), 1 with no segments (dropped)
    assert len(options) == 2
    assert options[0].stops == 1 and options[0].price_eur == 132


def test_text_fields_are_tidied():
    options = parse_serpapi(json.loads(FIXTURE.read_text()), QUERY)
    assert options[1].segments[0].airline == "Norwegian Air Shuttle"


def test_budget_is_a_hard_limit():
    budget = SearchBudget(2)
    budget.spend()
    budget.spend()
    with pytest.raises(BudgetExceeded):
        budget.spend()


def test_query_rejects_non_iata_codes():
    with pytest.raises(ValidationError):
        SearchQuery(origin="Stockholm", destination="VCE", day=date(2026, 12, 19))

