"""Build the hourly cell panel from parking events (data/processed/panel/).

    python scripts/build_panel.py [--config config/default.yaml]

Channels are saved as ``<name>.npy`` (int16, cells x hours), which can be memory-mapped for
training. ``cells.parquet`` and ``hours.parquet`` hold the row and column labels.
"""

import argparse
import time

import geopandas as gpd
import numpy as np
import pandas as pd

from oversupply.config import DEFAULT_CONFIG, Config
from oversupply.geo import Layers
from oversupply.panel import cell_centres, counts, read_event_arrays, stock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = Config.load(args.config)
    layers = Layers(cfg)
    out = cfg.paths.processed / "panel"
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    t0 = pd.Timestamp(cfg.study.start)
    hours = pd.date_range(t0, pd.Timestamp(cfg.study.end), freq="h", inclusive="left")
    n_hours = len(hours)
    size = cfg.panel.cell_size_m
    docking = layers.docking_stations.geometry.values
    ev = read_event_arrays(cfg.paths.processed / "parking_events.parquet", t0, size, docking,
                           cfg.panel.docking_buffer_m)
    keys, cell = np.unique(ev["key"], return_inverse=True)
    n_cells = len(keys)
    print(f"{len(cell):,} events; {n_cells:,} cells x {n_hours:,} hours", flush=True)

    outside = ~ev["near_dock"]
    channels = {
        "parked": lambda: stock(cell, ev["h_start"], ev["h_end"], n_cells, n_hours),
        "parked_outside_dock": lambda: stock(cell[outside], ev["h_start"][outside],
                                             ev["h_end"][outside], n_cells, n_hours),
        "arrivals": lambda: counts(cell, ev["h_start"], n_cells, n_hours),
        "departures": lambda: counts(cell, ev["h_end"], n_cells, n_hours),
        "provider_arrivals": lambda: counts(cell[ev["provider"]], ev["h_start"][ev["provider"]],
                                            n_cells, n_hours),
    }
    for name, compute in channels.items():
        np.save(out / f"{name}.npy", compute())
        print(f"  {name} saved ({time.perf_counter() - t_start:.0f}s)", flush=True)

    n_events = np.bincount(cell, minlength=n_cells)
    x, y = cell_centres(keys, size)
    cells = gpd.GeoDataFrame({
        "cell": np.arange(n_cells),
        "key": keys,
        "n_events": n_events,
        "dock_share": np.bincount(cell, ev["near_dock"], n_cells) / n_events,
        "provider_share": np.bincount(cell, ev["provider"], n_cells) / n_events,
    }, geometry=gpd.points_from_xy(x, y), crs=cfg.crs.projected)
    cells["is_depot"] = ((cells["n_events"] >= cfg.panel.depot_min_events)
                         & (cells["provider_share"] >= cfg.panel.depot_min_provider_share))
    # Model inputs may only know the training months; the depot rule above is a cleaning decision.
    in_training = ev["h_start"] < hours.searchsorted(pd.Timestamp(cfg.splits.tune_start))
    train_events = np.maximum(np.bincount(cell[in_training], minlength=n_cells), 1)
    cells["train_dock_share"] = (np.bincount(cell[in_training], ev["near_dock"][in_training],
                                             n_cells) / train_events)
    cells["train_provider_share"] = (np.bincount(cell[in_training], ev["provider"][in_training],
                                                 n_cells) / train_events)
    bike = gpd.sjoin_nearest(cells[["geometry"]], layers.bike_lanes[["geometry"]],
                             how="left", distance_col="dist")
    cells["dist_to_infra_m"] = bike.groupby(level=0)["dist"].first()
    dock = gpd.sjoin_nearest(cells[["geometry"]], layers.docking_stations[["geometry"]],
                             how="left", distance_col="dist")
    cells["dist_to_dock_m"] = dock.groupby(level=0)["dist"].first()
    area = gpd.sjoin(cells[["geometry"]], layers.areas, how="left", predicate="within")
    cells["area"] = area[~area.index.duplicated()]["area"]
    taz = gpd.sjoin(cells[["geometry"]], layers.taz, how="left", predicate="within")
    cells["TAZ"] = taz[~taz.index.duplicated()]["TAZ"]
    cells.to_parquet(out / "cells.parquet")
    pd.DataFrame({"time": hours}).to_parquet(out / "hours.parquet", index=False)
    print(f"done in {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
