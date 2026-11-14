from datetime import UTC, datetime, timedelta

import pytest

from fly_back_and_forth.storage import FlightStore


class Clock:
    """Controllable clock: tests move time forward instead of sleeping."""

    def __init__(self) -> None:
        self.now = datetime(2026, 11, 14, 9, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(tmp_path, clock) -> FlightStore:
    return FlightStore(tmp_path / "var" / "test.sqlite", clock=clock)

