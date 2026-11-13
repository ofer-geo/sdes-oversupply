import numpy as np
import pandas as pd

from oversupply.flows import flow_adjacency, vehicle_moves


def events(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["device_id", "start", "end", "end_reason", "key"])
    day = pd.Timestamp("2024-02-01")
    df["start_time"] = day + pd.to_timedelta(df["start"], "h")
    df["end_time"] = day + pd.to_timedelta(df["end"], "h")
    return df


def test_only_short_rides_of_one_vehicle_between_different_cells_count():
    moves = vehicle_moves(events([
        ("A", 0, 1, "on_trip", 10),
        ("A", 1.5, 2, "on_trip", 20),      # 10 -> 20: a ride
        ("A", 2.5, 3, "on_trip", 20),      # 20 -> 20: same cell, no move
        ("A", 3.5, 4, "removed", 20),      # 20 -> 20 again
        ("A", 5, 6, "on_trip", 30),        # after `removed`: an operator moved it
        ("A", 20, 21, "on_trip", 40),      # 30 -> 40 after 14 h: too long
        ("B", 0, 1, "on_trip", 50),        # last event of A must not link to B
    ]), max_gap_h=2.0)
    assert moves.values.tolist() == [[10, 20]]


def test_flow_rows_hold_the_strongest_sources_of_each_destination():
    moves = pd.DataFrame({"source": [1] * 9 + [2] * 6 + [3] * 2 + [4] * 7,
                          "destination": [7] * 17 + [8] * 7})
    keys = np.array([1, 2, 3, 4, 7, 8])
    a = flow_adjacency(moves, keys, top_sources=1, min_rides=5).toarray()
    assert a[4].tolist() == [9, 0, 0, 0, 0, 0]    # into 7: 1 beats 2; 3 is below the minimum
    assert a[5].tolist() == [0, 0, 0, 7, 0, 0]
    assert a[:4].sum() == 0
