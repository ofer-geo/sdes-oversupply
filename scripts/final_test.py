"""The final test on Nov-Dec 2024. Nothing here changes a model or a choice.

    python scripts/final_test.py

Uses the choices frozen on the tuning period (data/processed/final/choices.json) and the event list
locked before the test was opened (data/external/events_2024_nov_dec.csv). Writes
data/processed/final/test_metrics.csv, test_situations.csv and test_events.csv.

Event rules, fixed in advance: cells whose centre is within 500 m of the venue, from 2 hours
before the start hour to 3 hours after. Missile-alert dates (tier C) cover every cell for the
whole date. Unusual hour, which does not use the event list: a cell at 20+ and at least double
its usual level for that weekday and hour, where "usual" comes from training months only.
"""

import argparse
import json

import geopandas as gpd
import numpy as np
import pandas as pd

from oversupply.config import DEFAULT_CONFIG, REPO_ROOT, Config
from oversupply.data import load_model_data
from oversupply.metrics import K_BY_LEVEL, average_precision, blend, evaluate, top_flags

EVENTS = REPO_ROOT / "data/external/events_2024_nov_dec.csv"
RADIUS_M = 500
BEFORE_H, AFTER_H = 2, 3
UNUSUAL_LEVEL, UNUSUAL_RATIO = 20, 2.0
SELECTION_HORIZONS = (3, 6)
LEVEL = 20
CUTOFF_NAMES = {"LightGBM": "LightGBM", "GNN + flows": "GNN + flows (mean of seeds)",
                "Blend": "Blend"}


def event_masks(events: pd.DataFrame, centres: gpd.GeoSeries, times: pd.Series):
    """Cells x target-hours masks per tier, plus one mask per event."""
    masks = {tier: np.zeros((len(centres), len(times)), bool) for tier in "ABC"}
    venues = gpd.GeoSeries(gpd.points_from_xy(events["lon"], events["lat"]),
                           crs=4326).to_crs(centres.crs)
    per_event = []
    for i, e in events.iterrows():
        if e["tier"] == "C":
            hours = (times.dt.normalize() == pd.Timestamp(e["date"])).to_numpy()
            cells = np.ones(len(centres), bool)
        else:
            start = pd.Timestamp(f"{e['date']} {e['start']}").floor("h")
            hours = ((times >= start - pd.Timedelta(hours=BEFORE_H))
                     & (times <= start + pd.Timedelta(hours=AFTER_H))).to_numpy()
            cells = (centres.distance(venues.iloc[i]) <= RADIUS_M).to_numpy()
        mask = np.outer(cells, hours)
        masks[e["tier"]] |= mask
        per_event.append((e, mask))
    return masks, per_event


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = Config.load(args.config)
    data = load_model_data(cfg)
    panel, p = data.panel, data.panel.parked
    targets = np.arange(data.test_lo, data.test_hi)
    times = data.times.iloc[targets].reset_index(drop=True)
    y = p[:, targets].astype(np.float32)
    usual = panel.profile[:, panel.slot[targets]].astype(np.float32)
    horizons = cfg.forecast.horizons
    levels = tuple(cfg.forecast.levels)
    final = cfg.paths.processed / "final"
    choices = json.loads((final / "choices.json").read_text(encoding="utf-8"))
    folder = cfg.paths.processed / "predictions"
    print(f"test hours {len(targets):,}: {times.iloc[0]} to {times.iloc[-1]}", flush=True)

    with np.load(folder / "test_LightGBM.npz") as f:
        light = {k: f[k] for k in f.files}
    seeds = []
    for name in choices["gnn_checkpoints"]:
        with np.load(folder / f"test_{name}.npz") as f:
            seeds.append({k: f[k] for k in f.files})
    gnn = {h: np.mean([s[f"h{h}"] for s in seeds], axis=0) for h in horizons}
    weight = choices["blend_weight_lightgbm"]

    def forecasts(h: int) -> dict[str, np.ndarray]:
        return {"stays the same": p[:, targets - h].astype(np.float32),
                "same hour last week": p[:, targets - 168].astype(np.float32),
                "usual level": usual,
                "LightGBM": light[f"h{h}"], "GNN + flows": gnn[h],
                "Blend": blend(light[f"h{h}"], gnn[h], weight)}

    # 1. Overall accuracy
    rows = []
    for h in horizons:
        origin = p[:, targets - h]
        for name, pred in forecasts(h).items():
            row = {"horizon_h": h, "model": name, **evaluate(y, pred, origin, levels, K_BY_LEVEL)}
            if name in CUTOFF_NAMES:
                for level in levels:
                    cutoff = choices["cutoffs"][CUTOFF_NAMES[name]][str(h)][str(level)]
                    flagged, positive = pred >= cutoff, y >= level
                    hits = int((flagged & positive).sum())
                    row[f"cutoff_{level}"] = cutoff
                    row[f"cutoff_precision_{level}"] = hits / max(int(flagged.sum()), 1)
                    row[f"cutoff_recall_{level}"] = hits / max(int(positive.sum()), 1)
                    row[f"cutoff_f1_{level}"] = 2 * hits / max(
                        int(flagged.sum() + positive.sum()), 1)
            rows.append(row)
    metrics = pd.DataFrame(rows)
    final.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(final / "test_metrics.csv", index=False)

    print("\nSelection score (mean AP 20+ at 3 and 6 h):")
    score = (metrics[metrics["horizon_h"].isin(SELECTION_HORIZONS)]
             .groupby("model", sort=False)["ap_20"].mean().sort_values(ascending=False))
    print(score.round(4).to_string())
    k = K_BY_LEVEL[LEVEL]
    cols = ["mae", "ap_10", "ap_20", "ap_40", f"recall_top{k}_20", f"new_recall_top{k}_20",
            f"new_precision_top{k}_20", "cutoff_precision_20", "cutoff_recall_20", "cutoff_f1_20"]
    for h in horizons:
        print(f"\n{h} h ahead")
        print(metrics[metrics["horizon_h"] == h].set_index("model")[cols].round(3).to_string())

    # 2. Do LightGBM and the GNN find the same new pile-ups?
    print(f"\nNew {LEVEL}+ pile-ups found by each model ({k} flags per hour)")
    for h in horizons:
        below = p[:, targets - h] < LEVEL
        new = (y >= LEVEL) & below
        a = top_flags(light[f"h{h}"], below, k) & new
        b = top_flags(gnn[h], below, k) & new
        total = max(int(new.sum()), 1)
        print(f"  {h} h: new {total:,}; both {100 * (a & b).sum() / total:.1f}%, "
              f"LightGBM only {100 * (a & ~b).sum() / total:.1f}%, "
              f"GNN only {100 * (b & ~a).sum() / total:.1f}%, "
              f"neither {100 * (new & ~a & ~b).sum() / total:.1f}%", flush=True)

    # 3. Events and unusual hours
    cells = gpd.read_parquet(cfg.paths.processed / "panel" / "cells.parquet").set_index("key")
    centres = gpd.GeoSeries(cells.loc[data.keys].geometry.to_numpy(), crs=cells.crs)
    events = pd.read_csv(EVENTS, dtype={"start": str})
    masks, per_event = event_masks(events, centres, times)
    c_dates = masks["C"].any(axis=0)
    groups = {"A events": masks["A"], "B events": masks["B"], "C missile-alert dates": masks["C"],
              "no event": ~(masks["A"] | masks["B"] | masks["C"])}
    crowded = y >= LEVEL
    unusual = crowded & (y >= UNUSUAL_RATIO * usual)
    print(f"\nUnusual hours (20+ and at least double the usual): {int(unusual.sum()):,} of "
          f"{int(crowded.sum()):,} cell-hours at 20+. "
          f"In A windows {int((unusual & masks['A']).sum())}, "
          f"B windows {int((unusual & masks['B']).sum())}, on C dates "
          f"{int((unusual & masks['C']).sum())}.")

    situation_rows, event_rows = [], []
    learned = ["LightGBM", "GNN + flows", "Blend", "usual level", "stays the same"]
    for h in horizons:
        origin = p[:, targets - h]
        below = origin < LEVEL
        new = crowded & below
        f = forecasts(h)
        flags = {m: top_flags(f[m], np.ones_like(crowded), k) for m in learned}
        new_flags = {m: top_flags(f[m], below, k) for m in learned}
        for group, mask in groups.items():
            base = {"horizon_h": h, "group": group, "cell_hours": int(mask.sum()),
                    "actual_over_usual": float(y[mask].sum() / max(usual[mask].sum(), 1e-9)),
                    "n_20": int((crowded & mask).sum()), "n_new_20": int((new & mask).sum()),
                    "n_unusual": int((unusual & mask).sum())}
            for m in learned:
                pos, nw, un = crowded & mask, new & mask, unusual & mask
                situation_rows.append({
                    **base, "model": m,
                    "recall_20": (flags[m] & pos).sum() / max(int(pos.sum()), 1),
                    "recall_new_20": (new_flags[m] & nw).sum() / max(int(nw.sum()), 1),
                    "recall_unusual": (flags[m] & un).sum() / max(int(un.sum()), 1),
                    "ap_20_within": (average_precision(f[m][mask], pos[mask])
                                     if pos.sum() else float("nan")),
                    "forecast_share_of_crowd": float(f[m][pos].sum() / max(y[pos].sum(), 1e-9)),
                })
        for e, mask in per_event:
            if e["tier"] == "C":
                continue
            pos = crowded & mask
            event_rows.append({
                "horizon_h": h, "date": e["date"], "start": e["start"], "tier": e["tier"],
                "place": e["place"], "event": e["event"], "cells": int(mask.any(axis=1).sum()),
                "max_parked": int(y[mask].max()) if mask.any() else 0,
                "actual_over_usual": float(y[mask].sum() / max(usual[mask].sum(), 1e-9)),
                "n_20": int(pos.sum()), "n_new_20": int((new & mask).sum()),
                "n_unusual": int((unusual & mask).sum()),
                **{f"flagged_{m}": int((flags[m] & pos).sum())
                   for m in ("LightGBM", "GNN + flows", "Blend")},
            })
    situations = pd.DataFrame(situation_rows)
    situations.to_csv(final / "test_situations.csv", index=False)
    per_event_table = pd.DataFrame(event_rows)
    per_event_table.to_csv(final / "test_events.csv", index=False)
    print(f"C dates cover {int(c_dates.sum())} hours.")

    show = ["cell_hours", "actual_over_usual", "n_20", "n_new_20", "n_unusual", "recall_20",
            "recall_new_20", "recall_unusual", "ap_20_within", "forecast_share_of_crowd"]
    for h in horizons:
        print(f"\nSituations, {h} h ahead ({k} flags per hour, city-wide)")
        print(situations[situations["horizon_h"] == h].set_index(["group", "model"])[show]
              .round(3).to_string())
    print("\nEach event, 3 h ahead")
    print(per_event_table[per_event_table["horizon_h"] == 3].drop(columns="horizon_h")
          .round(2).to_string(index=False))


if __name__ == "__main__":
    main()
