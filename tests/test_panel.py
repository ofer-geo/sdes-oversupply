import numpy as np
import pandas as pd

from oversupply.panel import cell_centres, cell_key, counts, hour_index, stock

T0 = pd.Timestamp("2024-01-01")


def hours(*times: str) -> np.ndarray:
    return hour_index(pd.Series(pd.to_datetime(list(times))), T0)


def test_vehicle_counts_at_snapshots_from_start_until_before_end():
    start = hours("2024-01-01 00:30", "2024-01-01 02:00")
    end = hours("2024-01-01 03:00", "2024-01-01 02:30")
    m = stock(np.array([0, 0]), start, end, n_cells=1, n_hours=5)
    #  00:00 none; 01:00 and 02:00 first vehicle; 02:00 second vehicle too; 03:00 both gone
    assert m[0].tolist() == [0, 1, 2, 0, 0]


def test_short_stay_between_snapshots_is_not_counted():
    start, end = hours("2024-01-01 00:10"), hours("2024-01-01 00:50")
    assert stock(np.array([0]), start, end, 1, 3).sum() == 0


def test_flows_reconcile_with_stock():
    rng = np.random.default_rng(0)
    n, n_cells, n_hours = 500, 7, 48
    cell = rng.integers(0, n_cells, n)
    h_start = rng.integers(0, n_hours, n).astype(np.int32)
    h_end = h_start + rng.integers(0, 12, n).astype(np.int32)
    parked = stock(cell, h_start, h_end, n_cells, n_hours)
    arrivals = counts(cell, h_start, n_cells, n_hours)
    departures = counts(cell, h_end, n_cells, n_hours)
    assert (parked >= 0).all()
    assert np.array_equal(parked[:, 1:], parked[:, :-1] + arrivals[:, 1:] - departures[:, 1:])


def test_vehicle_still_parked_at_the_end_of_the_range_is_kept():
    m = stock(np.array([0]), np.array([1]), np.array([99]), 1, 3)
    assert m[0].tolist() == [0, 1, 1]


def test_cell_key_round_trips_to_the_cell_centre():
    x, y = np.array([178_123.0, 178_199.9]), np.array([664_001.0, 664_099.0])
    key = cell_key(x, y, 100.0)
    assert key[0] == key[1]
    cx, cy = cell_centres(key, 100.0)
    assert (cx[0], cy[0]) == (178_150.0, 664_050.0)
