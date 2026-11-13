"""Hourly grid-cell panel of parked scooters, built from parking events.

Every channel is a ``cells x hours`` matrix. A snapshot is taken at each full hour ``h``:

* ``parked``: vehicles with ``start <= h < end``, wherever they stand in the cell.
* ``parked_outside_dock``: the same, for vehicles away from any docking station.
* ``arrivals`` / ``departures``: events that started / ended in the hour *before* the snapshot,
  i.e. in ``(h - 1, h]``. They are known at time ``h`` and can be used as model inputs without
  leaking the future, and ``parked[h] = parked[h - 1] + arrivals[h] - departures[h]``.
* ``provider_arrivals``: the arrivals placed by an operator rather than left by a rider.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely

PROVIDER_DROP_OFF = "provider_drop_off"
_CELL_KEY = 1_000_000


def hour_index(times: pd.Series, t0: pd.Timestamp) -> np.ndarray:
    """Index of the first full-hour snapshot at or after each time."""
    return np.ceil((times - t0).dt.total_seconds().to_numpy() / 3600).astype(np.int32)


def cell_key(x: np.ndarray, y: np.ndarray, size: float) -> np.ndarray:
    return (np.floor(x / size).astype(np.int64) * _CELL_KEY
            + np.floor(y / size).astype(np.int64))


def cell_centres(keys: np.ndarray, size: float) -> tuple[np.ndarray, np.ndarray]:
    ix = (keys + _CELL_KEY // 2) // _CELL_KEY
    iy = keys - ix * _CELL_KEY
    return (ix + 0.5) * size, (iy + 0.5) * size


def counts(cells: np.ndarray, hours: np.ndarray, n_cells: int, n_hours: int) -> np.ndarray:
    """Events per cell and hour; events falling outside the time range are ignored."""
    inside = (hours >= 0) & (hours < n_hours)
    matrix = np.zeros((n_cells, n_hours), dtype=np.int16)
    np.add.at(matrix, (cells[inside], hours[inside]), 1)
    return matrix


def stock(cells: np.ndarray, h_start: np.ndarray, h_end: np.ndarray,
          n_cells: int, n_hours: int) -> np.ndarray:
    """Vehicles standing at each snapshot: a vehicle counts for ``h_start <= h < h_end``."""
    first = np.clip(h_start, 0, n_hours)
    last = np.clip(h_end, 0, n_hours)
    delta = np.zeros((n_cells, n_hours + 1), dtype=np.int16)
    np.add.at(delta, (cells, first), 1)
    np.add.at(delta, (cells, last), -1)
    return np.cumsum(delta, axis=1, dtype=np.int16)[:, :n_hours]


def read_event_arrays(path: str | Path, t0: pd.Timestamp, size: float,
                      docking: shapely.Geometry | list, buffer_m: float) -> dict[str, np.ndarray]:
    """Read only what the panel needs from the events file, one batch at a time."""
    tree = shapely.STRtree(docking)
    out: dict[str, list] = {k: [] for k in ("key", "h_start", "h_end", "near_dock", "provider")}
    file = pq.ParquetFile(path)
    columns = ["start_time", "end_time", "start_reason", "geometry"]
    for batch in file.iter_batches(batch_size=500_000, columns=columns):
        points = shapely.from_wkb(batch.column("geometry").to_numpy(zero_copy_only=False))
        near = np.zeros(len(points), dtype=bool)
        near[tree.query(points, predicate="dwithin", distance=buffer_m)[0]] = True
        reason = batch.column("start_reason").to_pandas().astype(str)
        out["key"].append(cell_key(shapely.get_x(points), shapely.get_y(points), size))
        out["h_start"].append(hour_index(batch.column("start_time").to_pandas(), t0))
        out["h_end"].append(hour_index(batch.column("end_time").to_pandas(), t0))
        out["near_dock"].append(near)
        out["provider"].append(reason.str.contains(PROVIDER_DROP_OFF, regex=False).to_numpy())
    return {k: np.concatenate(v) for k, v in out.items()}
