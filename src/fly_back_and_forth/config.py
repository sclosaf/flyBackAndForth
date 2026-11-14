"""Settings from environment (.env) and the search rules (config/search.yaml).

Everything the app writes stays inside the project folder:
- .env is read from the project root, not from the current directory
- the database lives in <project>/var/ by default
- a data dir outside the project (absolute path, "..", or a symlink pointing
  elsewhere) is rejected, so the app never writes to ~/.cache, /tmp or similar
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = _find_project_root()
DB_FILENAME = "flyBackAndForth.sqlite"

def _find_project_root() -> Path:
    root = Path(__file__).resolve().parents[2]

    if not (root / "pyproject.toml").is_file():
        raise RuntimeError("flyBackAndForth must run from its project folder (install with `uv pip install -e .`), so that its data stays inside the project")

    return root

def resolve_inside_project(path: Path, root: Path = PROJECT_ROOT) -> Path:
    resolved = (root / path).resolve()

    if not resolved.is_relative_to(root):
        raise ValueError(f"{path} resolves to {resolved}, outside the project {root}")

    return resolved

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    serpapi_api_key: SecretStr | None = None
    fly_back_and_forth_model: str = "openai:gpt-4o-mini"
    fly_back_and_forth_provider: str = "fake"
    fly_back_and_forth_max_searches: int = Field(default=6, ge=1, le=50)
    fly_back_and_forth_cache_ttl_hours: float = Field(default=6, ge=0, le=24 * 7)
    fly_back_and_forth_data_dir: Path = Path("var")

    @field_validator("fly_back_and_forth_data_dir")
    @classmethod
    def _inside_project(cls, value: Path) -> Path:
        return resolve_inside_project(value)

    @property
    def db_path(self) -> Path:
        return self.fly_back_and_forth_data_dir / DB_FILENAME


class AirportAccess(BaseModel):
    """Cost and time to reach or leave an airport (e.g. a far-away low-cost airport)."""

    cost_eur: float = Field(default=0, ge=0)
    minutes: int = Field(default=0, ge=0)
    note: str = ""


class SearchConfig(BaseModel):
    """Rules decided by you, not by the LLM: see config/search.yaml."""

    default_window_days: int = Field(default=7, ge=1, le=14)
    max_airports_per_side: int = Field(default=3, ge=1, le=6)
    value_of_time_eur_per_hour: float = Field(default=10, ge=0)
    top_n: int = Field(default=5, ge=1, le=50)
    metro_areas: dict[str, list[str]] = {}
    airport_access: dict[str, AirportAccess] = {}

    def expand(self, codes: list[str]) -> list[str]:
        airports: list[str] = []
        for code in codes:
            for airport in self.metro_areas.get(code, [code]):
                if airport not in airports:
                    airports.append(airport)

        return airports


def load_search_config(path: Path | None = None) -> SearchConfig:
    path = path or PROJECT_ROOT / "config" / "search.yaml"

    with open(path, encoding="utf-8") as f:
        return SearchConfig.model_validate(yaml.safe_load(f) or {})
