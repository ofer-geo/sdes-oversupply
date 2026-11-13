"""Typed project configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"


@dataclass(frozen=True)
class Paths:
    raw_status_changes: Path
    interim: Path
    processed: Path


@dataclass(frozen=True)
class Layers:
    city_boundary: Path | None
    taz: Path | None
    areas: Path | None
    bike_lanes: Path | None
    docking_stations: Path | None
    neighbourhoods: Path | None
    open_space: Path | None


@dataclass(frozen=True)
class CRS:
    geographic: str
    projected: str


@dataclass(frozen=True)
class Study:
    start: str
    end: str
    vehicle_type: str


@dataclass(frozen=True)
class Parking:
    same_location_m: float
    merge_gap_s: float
    low_battery_window_s: float
    min_duration_s: float
    open_space_buffer_m: float


@dataclass(frozen=True)
class Partitioning:
    n_buckets: int


@dataclass(frozen=True)
class Panel:
    cell_size_m: float
    docking_buffer_m: float
    depot_min_events: int
    depot_min_provider_share: float


@dataclass(frozen=True)
class Splits:
    warmup_end: str
    train_start: str
    tune_start: str
    test_start: str
    test_end: str


@dataclass(frozen=True)
class Forecast:
    horizons: list[int]
    max_horizon: int
    levels: list[int]
    min_active_peak: int


@dataclass(frozen=True)
class Weather:
    latitude: float
    longitude: float


@dataclass(frozen=True)
class Config:
    paths: Paths
    layers: Layers
    crs: CRS
    study: Study
    parking: Parking
    partitioning: Partitioning
    panel: Panel
    splits: Splits
    forecast: Forecast
    weather: Weather

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG) -> Config:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        kwargs = {}
        for f in fields(cls):
            section_cls = globals()[f.type]
            values = raw[f.name]
            if section_cls in (Paths, Layers):
                values = {k: _resolve(v) if v else None for k, v in values.items()}
            kwargs[f.name] = section_cls(**values)
        return cls(**kwargs)


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()
