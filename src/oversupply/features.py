"""Model inputs for forecasting parked scooters per cell.

Forecasts are made at an origin hour ``t`` for the target hour ``t + h``. To avoid leaking the
future, everything observed comes from ``t`` or earlier, including the weather. Only the
calendar, which is known in advance, describes the target hour itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import holidays
import numpy as np
import pandas as pd
import scipy.sparse as sp

SCHOOL_SUMMER_MONTHS = [7, 8]
MAX_ALLOWED_HORIZON = 24


def calendar_frame(times: pd.Series) -> pd.DataFrame:
    """Hour, weekday, Israeli public holidays, their eves and the school summer break."""
    times = pd.Series(pd.to_datetime(times)).reset_index(drop=True)
    years = sorted(times.dt.year.unique().tolist())
    days = {pd.Timestamp(d) for d in holidays.country_holidays("IL", years=years)}
    date = times.dt.normalize()
    summer = times.dt.month.isin(SCHOOL_SUMMER_MONTHS)
    return pd.DataFrame({
        "hour": times.dt.hour.astype(np.int8),
        "dow": times.dt.day_of_week.astype(np.int8),
        "month": times.dt.month.astype(np.int8),
        "is_holiday": date.isin(days).astype(np.int8),
        "is_holiday_eve": (date + pd.Timedelta(days=1)).isin(days).astype(np.int8),
        "school_summer": summer.astype(np.int8),
    })


def align_weather(weather: pd.DataFrame, times: pd.Series) -> pd.DataFrame:
    """Weather on the panel's hours. Clock changes create a repeated and a missing local hour."""
    w = weather.drop_duplicates("time").set_index("time").sort_index()
    w = w.reindex(pd.to_datetime(times)).interpolate(limit=3).ffill().bfill()
    return w.reset_index(drop=True)


def grid_adjacency(keys: np.ndarray, key_base: int = 1_000_000) -> sp.csr_matrix:
    """Sparse matrix linking each cell to its (up to 8) touching cells."""
    index = {int(k): i for i, k in enumerate(keys)}
    rows, cols = [], []
    for i, key in enumerate(keys):
        ix = (int(key) + key_base // 2) // key_base
        iy = int(key) - ix * key_base
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                j = index.get((ix + dx) * key_base + iy + dy)
                if (dx or dy) and j is not None:
                    rows.append(i)
                    cols.append(j)
    n = len(keys)
    return sp.csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n, n))


def weekly_profile(parked: np.ndarray, dow: np.ndarray, hour: np.ndarray,
                   mask: np.ndarray) -> np.ndarray:
    """Mean parked count per cell, weekday and hour (cells x 168), from ``mask`` hours only."""
    slot = dow.astype(np.int32) * 24 + hour
    sums = np.zeros((parked.shape[0], 168), dtype=np.float64)
    seen = np.zeros(168, dtype=np.int64)
    for t in np.flatnonzero(mask):
        sums[:, slot[t]] += parked[:, t]
        seen[slot[t]] += 1
    return (sums / np.maximum(seen, 1)).astype(np.float32)


@dataclass
class Panel:
    """Matrices (cells x hours) restricted to modelled cells, with their labels."""
    parked: np.ndarray
    arrivals: np.ndarray
    departures: np.ndarray
    provider: np.ndarray
    neighbour_parked: np.ndarray
    neighbour_arrivals: np.ndarray
    statics: pd.DataFrame
    calendar: pd.DataFrame
    weather: pd.DataFrame
    profile: np.ndarray

    def __post_init__(self) -> None:
        self.city_parked = self.parked.sum(axis=0, dtype=np.int32)
        self.slot = (self.calendar["dow"].to_numpy().astype(np.int32) * 24
                     + self.calendar["hour"].to_numpy())

    def rows(self, cell: np.ndarray, t: np.ndarray, h: np.ndarray) -> pd.DataFrame:
        """Features for forecasts made at hour ``t`` for ``t + h``. Requires ``t >= 168``."""
        if h.max() > MAX_ALLOWED_HORIZON:
            raise ValueError("Horizon too long: the daily lag would look into the future")
        if t.min() < 168:
            raise ValueError("Origin too early for the weekly lag")
        p, a, d = self.parked, self.arrivals, self.departures
        target = t + h
        out = {
            "parked": p[cell, t],
            "parked_1h": p[cell, t - 1],
            "parked_2h": p[cell, t - 2],
            "parked_3h": p[cell, t - 3],
            "parked_6h": p[cell, t - 6],
            "parked_24h": p[cell, t - 24],
            "target_yesterday": p[cell, target - 24],
            "target_last_week": p[cell, target - 168],
            "arrivals_1h": a[cell, t],
            "arrivals_3h": a[cell, t] + a[cell, t - 1] + a[cell, t - 2],
            "departures_1h": d[cell, t],
            "departures_3h": d[cell, t] + d[cell, t - 1] + d[cell, t - 2],
            "provider_3h": (self.provider[cell, t] + self.provider[cell, t - 1]
                            + self.provider[cell, t - 2]),
            "neighbour_parked": self.neighbour_parked[cell, t],
            "neighbour_arrivals_3h": (self.neighbour_arrivals[cell, t]
                                      + self.neighbour_arrivals[cell, t - 1]
                                      + self.neighbour_arrivals[cell, t - 2]),
            "city_parked": self.city_parked[t],
            "profile_target": self.profile[cell, self.slot[target]],
            "profile_now": self.profile[cell, self.slot[t]],
            "horizon": h,
        }
        for col in self.calendar.columns:
            out[f"target_{col}"] = self.calendar[col].to_numpy()[target]
        for col in self.weather.columns:
            out[col] = self.weather[col].to_numpy()[t]
        for col in self.statics.columns:
            out[col] = self.statics[col].to_numpy()[cell]
        return pd.DataFrame(out, dtype=np.float32)
