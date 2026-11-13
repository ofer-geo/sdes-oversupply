"""Parking events from MDS Status Changes.

The rules come from the thesis pipeline (github.com/ofer-geo/sdes-cycling-infrastructure), with
two differences, because this project uses Status Changes only:

* Events are not matched to rides in the Trips data, so the thesis filter "ride end without a
  matching trip" is not applied.
* Records with a malformed ``device_id`` (not a 36-character UUID) are removed.

A *parking event* is the time a vehicle stood ready to ride at one spot, and how it ended.
Consecutive "ready" records (``available`` / ``reserved``) of the same vehicle within
``same_location_m`` of each other belong to one event. The event ends at the first non-ready
record (its state is the end reason) or when the vehicle re-appears elsewhere (``unknown``).
"""

from __future__ import annotations

import contextlib

import geopandas as gpd
import numpy as np
import pandas as pd

from oversupply.config import Config
from oversupply.geo import Layers
from oversupply.report import FilterReport

READY = ("available", "reserved")
END_REASON = {"on_trip": "on_trip", "non_operational": "non_operational",
              "removed": "removed", "unknown": "unknown", "elsewhere": "unknown"}
DEVICE_ID_LENGTH = 36


class ParkingEventBuilder:
    def __init__(self, cfg: Config, layers: Layers | None):
        self.cfg = cfg
        self.rules = cfg.parking
        self.layers = layers
        self.report = FilterReport("Parking events")

    def run(self, status: pd.DataFrame) -> gpd.GeoDataFrame:
        """Build events from records holding the complete history of their vehicles."""
        events = self.build_events(self.prepare(status))
        events = self.drop_relocated(events)
        events = self.merge_interruptions(events)
        events = self.add_spatial_attributes(events)
        return self.drop_artifacts(events)

    def prepare(self, status: pd.DataFrame) -> gpd.GeoDataFrame:
        """Scooters with a valid id only, de-duplicated, projected, sorted per vehicle."""
        df = status[status["vehicle_type"] == self.cfg.study.vehicle_type]
        df = df[df["device_id"].str.len().eq(DEVICE_ID_LENGTH).fillna(False).astype(bool)]
        df = df.drop_duplicates(["device_id", "event_time", "vehicle_state", "lat", "lon"])
        df = df.sort_values(["device_id", "event_time"], kind="stable", ignore_index=True)
        geom = gpd.points_from_xy(df["lon"], df["lat"], crs=self.cfg.crs.geographic)
        return gpd.GeoDataFrame(df[["device_id", "vehicle_state", "event_types", "event_time"]],
                                geometry=geom).to_crs(self.cfg.crs.projected)

    def build_events(self, rec: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        same_dev_prev = rec["device_id"].eq(rec["device_id"].shift()).fillna(False).astype(bool)
        x, y = rec.geometry.x.to_numpy(), rec.geometry.y.to_numpy()
        step = np.r_[np.inf, np.hypot(np.diff(x), np.diff(y))]
        ready = rec["vehicle_state"].isin(READY).to_numpy()

        continues = ready & np.r_[False, ready[:-1]] & same_dev_prev.to_numpy() & (
            step <= self.rules.same_location_m)
        starts = ready & ~continues
        event_id = np.where(ready, np.cumsum(starts), 0)

        # the record that ends each event is the one right after its last ready record
        last_of_event = ready & ~np.r_[continues[1:], False]
        enders = np.flatnonzero(last_of_event) + 1
        device = rec["device_id"].to_numpy()
        in_bounds = enders < len(rec)
        clipped = np.minimum(enders, len(rec) - 1)
        has_end = in_bounds & (device[clipped] == device[enders - 1])

        first = rec.loc[starts].reset_index(drop=True)
        n_events = len(first)
        self.report.start(n_events)
        events = gpd.GeoDataFrame({
            "device_id": first["device_id"],
            "first_state": first["vehicle_state"].astype(str),
            "start_reason": first["event_types"].astype(str),
            "start_time": first["event_time"],
            "n_records": np.bincount(event_id[ready], minlength=n_events + 1)[1:],
        }, geometry=first.geometry.values, crs=rec.crs)

        # an event without a following record (end of data) has no end; use a dummy row
        end_rec = rec.iloc[np.where(has_end, enders, 0)]
        end_state = end_rec["vehicle_state"].astype(str).to_numpy()
        events["end_time"] = end_rec["event_time"].to_numpy()
        events["end_x"] = end_rec.geometry.x.to_numpy()
        events["end_y"] = end_rec.geometry.y.to_numpy()
        events["end_reason"] = np.where(np.isin(end_state, READY), "unknown",
                                        pd.Series(end_state).map(END_REASON).fillna("unknown"))

        events = self._drop(events, ~has_end, "no following record (censored at end of data)")
        events = self._drop(events, events["first_state"] != "available",
                            "starts with an orphan reservation")
        events = self._drop(events, events["end_time"] < pd.Timestamp(self.cfg.study.start),
                            "ended before study period")
        return events.reset_index(drop=True)

    def drop_relocated(self, events: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """The vehicle should leave from where it was parked; a larger gap means the
        recorded location is unreliable."""
        gap = np.hypot(events["end_x"] - events.geometry.x, events["end_y"] - events.geometry.y)
        return self._drop(events, gap > self.rules.same_location_m,
                          f"start/end location > {self.rules.same_location_m:g} m apart")

    def merge_interruptions(self, events: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Merge an event with the next one of the same vehicle if the vehicle was 'away'
        only briefly and came back to the same spot (a ride without movement, a short
        operator status flip...)."""
        e = events.sort_values(["device_id", "start_time"], ignore_index=True)
        nxt = e.shift(-1)
        gap_s = (nxt["start_time"] - e["end_time"]).dt.total_seconds()
        dist = np.hypot(nxt.geometry.x - e.geometry.x, nxt.geometry.y - e.geometry.y)
        same_device = nxt["device_id"].eq(e["device_id"]).fillna(False).astype(bool)
        joins_next = (same_device & (gap_s <= self.rules.merge_gap_s)
                      & (dist <= self.rules.same_location_m))
        group = np.cumsum(~np.r_[False, joins_next.to_numpy()[:-1]])

        g = e.groupby(group, sort=False)
        merged = g.first()
        for col in ("end_time", "end_x", "end_y", "end_reason"):
            merged[col] = g[col].last()
        merged["n_records"] = g["n_records"].sum()
        self.report.add("merged with the next event (short interruption)",
                        len(e) - len(merged))
        return gpd.GeoDataFrame(merged.reset_index(drop=True), geometry="geometry", crs=e.crs)

    def add_spatial_attributes(self, events: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        layers = self.layers
        events = self._drop(events, ~events.within(layers.city_boundary), "outside city")
        try:
            open_space = layers.open_space.buffer(self.rules.open_space_buffer_m)
            events = self._drop(events, ~events.within(open_space),
                                "not in public open space (GPS error)")
        except FileNotFoundError:
            pass

        events = self._join_polygons(events, layers.areas, "area")
        events = self._join_polygons(events, layers.taz, "TAZ")
        with contextlib.suppress(FileNotFoundError):  # optional layer
            events = self._join_polygons(events, layers.neighbourhoods, None)
        nearest = gpd.sjoin_nearest(events[["geometry"]], layers.bike_lanes[["geometry"]],
                                    how="left", distance_col="dist_to_infra_m")
        events["dist_to_infra_m"] = nearest.groupby(level=0)["dist_to_infra_m"].first()

        events["duration_s"] = (events["end_time"] - events["start_time"]).dt.total_seconds()
        events["start_hour"] = events["start_time"].dt.hour
        events["end_hour"] = events["end_time"].dt.hour
        events["weekday"] = events["start_time"].dt.day_name()
        return events

    def drop_artifacts(self, events: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        r = self.rules
        low_battery = (events["end_reason"].isin(["non_operational", "removed"])
                       & (events["duration_s"] < r.low_battery_window_s))
        events = self._drop(events, low_battery,
                            f"taken out of service within {r.low_battery_window_s / 60:g} min "
                            "(low battery after a ride)")
        events = self._drop(events, events["duration_s"] < r.min_duration_s,
                            f"shorter than {r.min_duration_s:g} s")
        return events.reset_index(drop=True)

    def _drop(self, df, mask, step: str):
        mask = np.asarray(mask, dtype=bool)
        self.report.add(step, int(mask.sum()))
        return df.loc[~mask]

    @staticmethod
    def _join_polygons(points: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame,
                       column: str | None) -> gpd.GeoDataFrame:
        cols = [column, "geometry"] if column else list(polygons.columns)
        joined = gpd.sjoin(points, polygons[cols], how="left", predicate="within")
        joined = joined[~joined.index.duplicated()]
        return joined.drop(columns="index_right")
