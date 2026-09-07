"""City configuration loading.

A city is described by one YAML file in config/cities/. Nothing in the code
is specific to Monterrey: adding a city means adding a YAML file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CITY_DIR = REPO_ROOT / "config" / "cities"
DATA_DIR = Path(os.environ.get("BANQUETAS_DATA", REPO_ROOT / "data"))


@dataclass
class CityConfig:
    name: str
    label: str
    country: str
    bbox: tuple[float, float, float, float]  # west, south, east, north (EPSG:4326)
    inegi: dict[str, Any] = field(default_factory=dict)
    mapillary: dict[str, Any] = field(default_factory=dict)
    osm: dict[str, Any] = field(default_factory=dict)
    tiers: dict[str, list[str]] = field(default_factory=dict)
    camera: dict[str, Any] = field(default_factory=dict)
    crs_metric: str | None = None

    @property
    def camera_height_m(self) -> float:
        return float(self.camera.get("height_m", 1.3))

    @property
    def camera_pitch_deg(self) -> float:
        return float(self.camera.get("pitch_deg", 0.0))

    @property
    def raw_dir(self) -> Path:
        return _ensure(DATA_DIR / "raw" / self.name)

    @property
    def interim_dir(self) -> Path:
        return _ensure(DATA_DIR / "interim" / self.name)

    @property
    def out_dir(self) -> Path:
        return _ensure(DATA_DIR / "out" / self.name)

    @property
    def tile_cache(self) -> Path:
        return _ensure(DATA_DIR / "raw" / self.name / "mapillary_tiles")

    def tier_of(self, highway: str | None) -> str:
        """Map an OSM highway tag to a coarse hierarchy tier."""
        if not highway:
            return "other"
        for tier, tags in self.tiers.items():
            if highway in tags:
                return tier
        return "other"


def _ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_city(name: str) -> CityConfig:
    path = CITY_DIR / f"{name}.yml"
    if not path.exists():
        available = sorted(p.stem for p in CITY_DIR.glob("*.yml") if not p.stem.startswith("_"))
        raise FileNotFoundError(f"No config for '{name}'. Available: {', '.join(available)}")
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    raw["bbox"] = tuple(float(v) for v in raw["bbox"])
    return CityConfig(**raw)


def mapillary_token() -> str:
    """Read the Mapillary token from the environment.

    Never hardcode it and never commit it. Put it in a local .env file and
    export it, or set it in your shell profile.
    """
    token = os.environ.get("MAPILLARY_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "MAPILLARY_TOKEN is not set. Get a free client token at "
            "https://www.mapillary.com/dashboard/developers and export it, "
            "for example: export MAPILLARY_TOKEN='MLY|...'"
        )
    return token
