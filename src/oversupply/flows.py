"""A graph of where scooters travel, from consecutive parking events of the same vehicle.

The event data has no trips, but a vehicle that leaves one cell on a ride and soon parks in
another reveals a flow between those cells. Rides link distant cells, which the grid's touching
neighbours cannot do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def vehicle_moves(events: pd.DataFrame, max_gap_h: float) -> pd.DataFrame:
    """Source and destination keys of rides between consecutive events of each vehicle.

    ``events`` needs device_id, start_time, end_time, end_reason and key.
    """
    e = events.sort_values(["device_id", "start_time"], kind="stable")
    same_vehicle = e["device_id"].eq(e["device_id"].shift(-1)).fillna(False).astype(bool)
    gap = (e["start_time"].shift(-1) - e["end_time"]).dt.total_seconds() / 3600
    is_ride = (same_vehicle & (e["end_reason"] == "on_trip") & (gap >= 0)
               & (gap <= max_gap_h)).to_numpy()
    moved = e["key"].to_numpy()[:-1] != e["key"].to_numpy()[1:] if len(e) > 1 else np.array([])
    keep = is_ride[:-1] & moved if len(e) > 1 else np.zeros(0, dtype=bool)
    return pd.DataFrame({"source": e["key"].to_numpy()[:-1][keep],
                         "destination": e["key"].to_numpy()[1:][keep]})


def flow_adjacency(moves: pd.DataFrame, keys: np.ndarray, top_sources: int,
                   min_rides: int) -> sp.csr_matrix:
    """Matrix whose row i holds the cells that send scooters to cell i.

    Each row keeps the ``top_sources`` strongest links with at least ``min_rides`` rides.
    """
    index = pd.Series(np.arange(len(keys)), index=keys)
    source = moves["source"].map(index)
    destination = moves["destination"].map(index)
    links = pd.DataFrame({"i": destination, "j": source}).dropna().astype(np.int64)
    counts = links.groupby(["i", "j"]).size().rename("rides").reset_index()
    counts = counts[counts["rides"] >= min_rides]
    counts = (counts.sort_values(["i", "rides"], ascending=[True, False])
              .groupby("i").head(top_sources))
    return sp.csr_matrix((counts["rides"].to_numpy(np.float32),
                          (counts["i"].to_numpy(), counts["j"].to_numpy())),
                         shape=(len(keys), len(keys)))
