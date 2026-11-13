import numpy as np
import pandas as pd
import pytest

from oversupply.features import Panel, align_weather, calendar_frame, grid_adjacency, weekly_profile
from oversupply.metrics import average_precision, best_cutoff, blend, evaluate, top_k

N_CELLS, N_HOURS = 3, 400


def small_panel(seed: int = 0) -> Panel:
    rng = np.random.default_rng(seed)
    times = pd.Series(pd.date_range("2024-01-01", periods=N_HOURS, freq="h"))

    def m() -> np.ndarray:
        return rng.integers(0, 20, (N_CELLS, N_HOURS)).astype(np.int16)

    return Panel(m(), m(), m(), m(), m(), m(),
                 pd.DataFrame({"dock_share": rng.random(N_CELLS)}),
                 calendar_frame(times),
                 pd.DataFrame({"temperature_2m": rng.random(N_HOURS)}),
                 rng.random((N_CELLS, 168)).astype(np.float32))


def test_changing_the_future_does_not_change_features():
    panel = small_panel()
    cell, t = np.array([0, 1, 2]), np.array([200, 250, 300])
    h = np.array([6, 6, 6])
    before = panel.rows(cell, t, h)
    # Scramble every observed value after each origin, including the target itself
    for matrix in (panel.parked, panel.arrivals, panel.departures, panel.provider,
                   panel.neighbour_parked, panel.neighbour_arrivals):
        for c, origin in zip(cell, t, strict=True):
            matrix[c, origin + 1:] = 99
    panel.weather.iloc[301:, 0] = 99.0
    after = panel.rows(cell, t, h)
    changed = [c for c in before.columns if not before[c].equals(after[c])]
    # City total sums all cells, whose futures overlap other rows' pasts in this test
    assert set(changed) <= {"city_parked"}


def test_too_long_horizon_or_too_early_origin_is_refused():
    panel = small_panel()
    with pytest.raises(ValueError):
        panel.rows(np.array([0]), np.array([200]), np.array([25]))
    with pytest.raises(ValueError):
        panel.rows(np.array([0]), np.array([100]), np.array([1]))


def test_calendar_knows_yom_kippur_and_its_eve():
    c = calendar_frame(pd.Series(pd.to_datetime(["2024-10-11 12:00", "2024-10-12 12:00",
                                                 "2024-10-14 12:00", "2024-07-02 09:00"])))
    assert c["is_holiday"].tolist() == [0, 1, 0, 0]
    assert c["is_holiday_eve"].tolist()[:2] == [1, 0]
    assert c["school_summer"].tolist() == [0, 0, 0, 1]


def test_weather_fills_the_clock_change_gap_and_duplicate():
    times = pd.Series(pd.date_range("2024-03-29 00:00", periods=4, freq="h"))
    raw = pd.DataFrame({"time": pd.to_datetime(["2024-03-29 00:00", "2024-03-29 01:00",
                                                "2024-03-29 01:00", "2024-03-29 03:00"]),
                        "temperature_2m": [10.0, 12.0, 99.0, 16.0]})
    w = align_weather(raw, times)
    assert w["temperature_2m"].tolist() == [10.0, 12.0, 14.0, 16.0]


def test_adjacency_links_touching_cells_only():
    base = 1_000_000
    keys = np.array([10 * base + 10, 10 * base + 11, 11 * base + 11, 13 * base + 10])
    a = grid_adjacency(keys).toarray()
    assert a.tolist() == [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]]


def test_profile_uses_only_masked_hours():
    parked = np.array([[10, 99, 20, 99]], dtype=np.int16)
    dow, hour = np.zeros(4, int), np.array([0, 1, 0, 1])
    profile = weekly_profile(parked, dow, hour, np.array([True, False, True, False]))
    assert profile[0, 0] == 15.0
    assert profile[0, 1] == 0.0


def test_blend_is_a_weighted_geometric_mean():
    first, second = np.array([4.0, 0.0]), np.array([1.0, 0.0])
    assert blend(first, second, 1.0)[0] == pytest.approx(4.0)
    assert blend(first, second, 0.5)[0] == pytest.approx(2.0, abs=0.01)
    assert blend(first, second, 0.5)[1] == pytest.approx(0.0, abs=1e-6)


def test_best_cutoff_maximizes_f1():
    pred = np.array([9.0, 8.0, 7.0, 1.0])
    positive = np.array([True, True, False, False])
    cutoff, f1 = best_cutoff(pred, positive, np.array([0.5, 7.5, 8.5]))
    assert (cutoff, f1) == (7.5, 1.0)


def test_average_precision_is_one_for_perfect_ranking():
    score = np.array([0.9, 0.8, 0.1, 0.0])
    assert average_precision(score, np.array([True, True, False, False])) == 1.0


def test_top_k_counts_flags_per_hour():
    score = np.array([[3.0, 3.0], [2.0, 2.0], [1.0, 1.0]])
    positive = np.array([[True, False], [False, True], [True, False]])
    assert top_k(score, positive, 1) == (0.5, 1 / 3)


def test_new_pile_ups_exclude_cells_already_crowded():
    y = np.array([[30.0], [30.0], [1.0]])
    pred = np.array([[40.0], [25.0], [2.0]])
    origin = np.array([[30.0], [3.0], [1.0]])     # cell 0 was already crowded
    out = evaluate(y, pred, origin, (20,), {20: 1})
    assert out["new_share_20"] == 0.5
    assert out["new_precision_top1_20"] == 1.0    # cell 1 is flagged, not cell 0
    assert out["new_recall_top1_20"] == 1.0
