"""Load the panel for modelling: usable cells, inputs and the time splits."""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp

from oversupply.config import REPO_ROOT, Config
from oversupply.features import (
    Panel,
    align_weather,
    calendar_frame,
    grid_adjacency,
    weekly_profile,
)


@dataclass
class ModelData:
    panel: Panel
    times: pd.Series
    adjacency: sp.csr_matrix
    keys: np.ndarray
    train_lo: int
    tune_lo: int
    test_lo: int
    test_hi: int


def load_model_data(cfg: Config) -> ModelData:
    """Depot-like cells are excluded. Modelled cells and profiles use training hours only."""
    folder = cfg.paths.processed / "panel"
    cells = gpd.read_parquet(folder / "cells.parquet")
    times = pd.read_parquet(folder / "hours.parquet")["time"]
    usable = ~cells["is_depot"].to_numpy()
    parked_all = np.load(folder / "parked.npy", mmap_mode="r")

    train = ((times >= cfg.splits.warmup_end) & (times < cfg.splits.tune_start)).to_numpy()
    active = usable & (np.asarray(parked_all[:, train]).max(axis=1)
                       >= cfg.forecast.min_active_peak)
    print(f"cells: {len(cells):,}, depot-like {(~usable).sum()}, modelled {active.sum():,}",
          flush=True)
    parked = np.asarray(parked_all[active])
    arrivals = np.load(folder / "arrivals.npy", mmap_mode="r")[active].copy()
    departures = np.load(folder / "departures.npy", mmap_mode="r")[active].copy()
    provider = np.load(folder / "provider_arrivals.npy", mmap_mode="r")[active].copy()

    kept = cells[active].reset_index(drop=True)
    adjacency = grid_adjacency(kept["key"].to_numpy())
    calendar = calendar_frame(times)
    weather = align_weather(pd.read_parquet(REPO_ROOT / "data/external/weather.parquet"), times)
    profile = weekly_profile(parked, calendar["dow"].to_numpy(), calendar["hour"].to_numpy(), train)
    statics = pd.DataFrame({
        "cell_x": kept.geometry.x, "cell_y": kept.geometry.y,
        "dock_share": kept["train_dock_share"], "dist_to_dock_m": kept["dist_to_dock_m"],
        "dist_to_infra_m": kept["dist_to_infra_m"],
        "provider_share": kept["train_provider_share"],
    })
    panel = Panel(parked, arrivals, departures, provider,
                  (adjacency @ parked).astype(np.int16), (adjacency @ arrivals).astype(np.int16),
                  statics, calendar, weather, profile)

    def index(stamp: str) -> int:
        return int(np.searchsorted(times, pd.Timestamp(stamp)))

    return ModelData(panel, times, adjacency, kept["key"].to_numpy(),
                     index(cfg.splits.train_start),
                     index(cfg.splits.tune_start), index(cfg.splits.test_start),
                     index(cfg.splits.test_end))
