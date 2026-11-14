"""Data contracts between the LLM, the tools and the graph.

LLM output and API responses are parsed into these models, so malformed or
unexpected data fails here, with a clear error, instead of deep inside the graph.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from enum import Enum
from itertools import pairwise
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field, field_validator, model_validator

IATA = r"^[A-Z]{3}$"
MAX_WINDOW_DAYS = 14

LocationCode = Annotated[str, BeforeValidator(_normalise_code), Field(pattern=IATA)]

def _normalise_code(value: object) -> object:
    return value.strip().upper() if isinstance(value, str) else value


class BaggageNeed(str, Enum):
    PERSONAL_ITEM = "personal_item"
    CABIN = "cabin"
    CHECKED = "checked"

class Optimize(str, Enum):
    PRICE = "price"
    DURATION = "duration"
    VALUE = "value"

class TripRequest(BaseModel):
    """What the user wants, extracted by the LLM with structured output (or given as flags)."""

    origins: list[LocationCode] = Field(min_length=1, max_length=3, description="Departure airport or metropolitan area IATA codes, e.g. ['ARN'] or ['STO']")
    destinations: list[LocationCode] = Field(min_length=1, max_length=3, description="Arrival airport or metropolitan area IATA codes, e.g. ['VCE', 'TSF']")

    earliest_departure: date = Field(description="First acceptable departure date")
    latest_departure: date | None = Field(default=None, description="Last acceptable departure date. Empty if the user only gave a start date")

    max_price_eur: float | None = Field(default=None, gt=0, description="Budget, if given")
    max_stops: int | None = Field(default=None, ge=0, le=3, description="0 for direct flights only; empty if not said")
    optimize: Optimize = Field(default=Optimize.PRICE, description="price unless the user asks for fastest or a balance")
    baggage: BaggageNeed = Field(default=BaggageNeed.CABIN)
    notes: str = Field(default="", max_length=300, description="Other preferences, verbatim")

    @model_validator(mode="after")
    def check_route_and_dates(self) -> TripRequest:
        if set(self.origins) & set(self.destinations):
            raise ValueError("origin and destination are the same")
        if self.latest_departure is not None:
            if self.latest_departure < self.earliest_departure:
                raise ValueError("latest_departure is before earliest_departure")
            if (self.latest_departure - self.earliest_departure).days >= MAX_WINDOW_DAYS:
                raise ValueError(f"date window longer than {MAX_WINDOW_DAYS} days: narrow it down")
        return self

    def window(self, default_days: int) -> tuple[date, date]:
        """Departure dates to search; "from a date" means default_days starting that day."""
        end = self.latest_departure or self.earliest_departure + timedelta(days=default_days - 1)
        return self.earliest_departure, end

class SearchQuery(BaseModel):
    """One concrete search: one route, one day. Built by code, never by the LLM."""

    origin: str = Field(pattern=IATA)
    destination: str = Field(pattern=IATA)
    day: date

    @property
    def key(self) -> str:
        return f"{self.origin}-{self.destination}-{self.day.isoformat()}"


class Segment(BaseModel):
    origin: str = Field(pattern=IATA)
    destination: str = Field(pattern=IATA)
    departure: datetime
    arrival: datetime
    airline: str = Field(max_length=60)
    flight_number: str = Field(default="", max_length=12)

    @model_validator(mode="after")
    def arrives_after_departing(self) -> Segment:
        if self.arrival <= self.departure:
            raise ValueError("segment arrives before it departs")
        return self


class FlightOption(BaseModel):
    """A bookable itinerary as returned by a provider, validated on arrival."""

    query_key: str
    price_eur: float = Field(gt=0, lt=5000)
    segments: list[Segment] = Field(min_length=1, max_length=4)
    total_minutes: int = Field(gt=0)
    source: str

    @property
    def stops(self) -> int:
        return len(self.segments) - 1

    @property
    def departure(self) -> datetime:
        return self.segments[0].departure

    @property
    def origin(self) -> str:
        return self.segments[0].origin

    @property
    def destination(self) -> str:
        return self.segments[-1].destination

    @field_validator("segments")
    @classmethod
    def chronological(cls, segs: list[Segment]) -> list[Segment]:
        for a, b in pairwise(segs):
            if b.departure < a.arrival:
                raise ValueError("segments are not in chronological order")
        return segs


class RankedOption(BaseModel):
    """A flight plus everything around it: what the trip really costs, door to door."""

    flight: FlightOption
    access_cost_eur: float = 0
    baggage_cost_eur: float = 0
    door_to_door_minutes: int

    @property
    def total_cost_eur(self) -> float:
        return round(self.flight.price_eur + self.access_cost_eur + self.baggage_cost_eur, 2)
