"""Figures and result tables for the README, from the final test (Nov-Dec 2024).

    python scripts/make_figures.py

Needs matplotlib (pip install -e .[figures]). Reads data/processed/final/ and the saved test
forecasts; writes docs/figures/*.png and results/*.csv. Nothing here changes a model or a choice.
"""

import json
import shutil

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from oversupply.config import DEFAULT_CONFIG, REPO_ROOT, Config
from oversupply.data import load_model_data
from oversupply.geo import read_layer
from oversupply.metrics import K_BY_LEVEL, blend, top_flags

FIGURES = REPO_ROOT / "docs" / "figures"
RESULTS = REPO_ROOT / "results"
LEVEL, H = 20, 3

INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
SERIES = {"LightGBM": "#2a78d6", "GNN + flows": "#eb6834", "stays the same": "#1baf7a",
          "usual level": MUTED}
LABELS = {"LightGBM": "LightGBM (≈ blend)", "GNN + flows": "GNN + flows",
          "stays the same": "No change", "usual level": "Usual level"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def skill_chart(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    panels = [("ap_20", "Average precision, 20+ scooters per cell"),
              (f"new_recall_top{K_BY_LEVEL[LEVEL]}_20", "New pile-ups caught (10 flags/hour)")]
    for ax, (col, title) in zip(axes, panels, strict=True):
        for name, color in SERIES.items():
            rows = metrics[metrics["model"] == name].sort_values("horizon_h")
            dashed = name == "usual level"
            ax.plot(rows["horizon_h"], rows[col], color=color, lw=2,
                    ls="--" if dashed else "-", marker="o", ms=6,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=LABELS[name])
        ends = metrics[metrics["horizon_h"] == 6].set_index("model")[col]
        placed = []
        for name in sorted(SERIES, key=lambda n: -ends[n]):
            y_text = min([ends[name]] + [q - 0.045 for q in placed])
            placed.append(y_text)
            ax.annotate(f"{ends[name]:.2f}", (6, ends[name]), xytext=(6.18, y_text),
                        textcoords="data", va="center", fontsize=9, color=INK_2)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=10)
        ax.set_xticks([1, 3, 6], ["1 h", "3 h", "6 h"])
        ax.set_xlim(0.6, 6.9)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("Forecast horizon")
    axes[0].legend(frameon=False, loc="lower left", fontsize=9)
    fig.suptitle("Held-out test, Nov–Dec 2024: forecasting scooter pile-ups", x=0.01,
                 ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "test_skill.png", dpi=180)
    plt.close(fig)


def snapshot_map(cfg: Config) -> str:
    """One test hour: where 20+ scooters stood, and which cells the blend flagged 3 h earlier."""
    data = load_model_data(cfg)
    p = data.panel.parked
    targets = np.arange(data.test_lo, data.test_hi)
    times = data.times.iloc[targets].reset_index(drop=True)
    folder = cfg.paths.processed / "predictions"
    final = cfg.paths.processed / "final"
    choices = json.loads((final / "choices.json").read_text(encoding="utf-8"))
    with np.load(folder / "test_LightGBM.npz") as f:
        light = f[f"h{H}"]
    gnn = np.mean([np.load(folder / f"test_{n}.npz")[f"h{H}"]
                   for n in choices["gnn_checkpoints"]], axis=0)
    pred = blend(light, gnn, choices["blend_weight_lightgbm"])
    y = p[:, targets]
    origin = p[:, targets - H]
    flags = top_flags(pred, np.ones_like(y, bool), K_BY_LEVEL[LEVEL])

    # The Sun-Thu (Israeli work week) evening hour with the most new pile-ups.
    new = ((y >= LEVEL) & (origin < LEVEL)).sum(axis=0)
    weekday_evening = (times.dt.day_of_week.isin([6, 0, 1, 2, 3])
                       & times.dt.hour.between(17, 21)).to_numpy()
    t = int(np.argmax(np.where(weekday_evening, new, -1)))

    cells = gpd.read_parquet(cfg.paths.processed / "panel" / "cells.parquet").set_index("key")
    grid = cells.loc[data.keys].reset_index()
    grid["geometry"] = grid.geometry.buffer(cfg.panel.cell_size_m / 2, cap_style="square")
    grid["parked"], grid["flag"] = y[:, t], flags[:, t]
    grid["new"] = (y[:, t] >= LEVEL) & (origin[:, t] < LEVEL)
    city = read_layer(cfg.layers.city_boundary, cfg.crs.projected)
    lanes = read_layer(cfg.layers.bike_lanes, cfg.crs.projected)

    events = pd.read_csv(REPO_ROOT / "data/external/events_2024_nov_dec.csv", dtype={"start": str})
    day = events[events["date"] == f"{times.iloc[t]:%Y-%m-%d}"]
    venues = gpd.GeoDataFrame(day, geometry=gpd.points_from_xy(day["lon"], day["lat"]),
                              crs=4326).to_crs(cfg.crs.projected)

    fig, ax = plt.subplots(figsize=(8, 9.5))
    city.plot(ax=ax, facecolor="#f3f2ee", edgecolor="#c9c8c2", lw=1)
    grid.plot(ax=ax, facecolor="#e6e5e0", edgecolor=SURFACE, lw=0.4)
    lanes.plot(ax=ax, color="#c4c3bc", lw=0.7)
    crowded = grid[grid["parked"] >= LEVEL]
    ramp = plt.get_cmap("Blues")
    crowded.plot(ax=ax, column="parked", cmap=ramp, vmin=0, vmax=60, edgecolor=SURFACE, lw=0.4)
    hit = grid[grid["flag"]]
    hit.plot(ax=ax, facecolor="none", edgecolor="#eb6834", lw=2)
    focus = pd.concat([crowded.geometry, hit.geometry, venues.geometry])
    minx, miny, maxx, maxy = focus.total_bounds
    pad = 500
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    for _, v in venues.iterrows():
        ax.annotate(f"{v['place']}\n{v['event']}, {v['start']}", (v.geometry.x, v.geometry.y),
                    xytext=(45, -40), textcoords="offset points", fontsize=8.5, color=INK,
                    ha="left", arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.8},
                    bbox={"boxstyle": "round,pad=0.3", "fc": SURFACE, "ec": GRID})
    ax.set_axis_off()
    caught = int((grid["flag"] & (grid["parked"] >= LEVEL)).sum())
    stamp = times.iloc[t]
    ax.set_title(f"{stamp:%a %d %b %Y, %H:00}, forecast made 3 h earlier\n"
                 f"{len(crowded)} cells had 20+ scooters; {caught} of the model's 10 flags "
                 f"were among them", loc="left", fontsize=11)
    handles = [Patch(facecolor=ramp(0.7), label="20+ scooters parked (darker = more)"),
               Patch(facecolor="none", edgecolor="#eb6834", lw=2,
                     label="Flagged by the model (top 10 cells)"),
               Patch(facecolor="#e6e5e0", label="Modelled 100 m cells"),
               Line2D([], [], color="#c4c3bc", lw=1.5, label="Cycling infrastructure")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2,
              frameon=False, fontsize=9)
    fig.text(0.99, -0.03, "GIS layers: Tel Aviv-Yafo Municipality", ha="right", fontsize=7.5,
             color=MUTED)
    fig.tight_layout()
    fig.savefig(FIGURES / "test_snapshot_map.png", dpi=180, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return f"{stamp:%Y-%m-%d %H:00}"


def main() -> None:
    cfg = Config.load(DEFAULT_CONFIG)
    final = cfg.paths.processed / "final"
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name in ("choices.json", "test_metrics.csv", "test_situations.csv", "test_events.csv"):
        shutil.copy(final / name, RESULTS / name)
    shutil.copy(cfg.paths.processed / "benchmarks" / "benchmarks_tune.csv",
                RESULTS / "tuning_benchmarks.csv")
    shutil.copy(cfg.paths.processed / "parking_events_report.csv",
                RESULTS / "parking_events_funnel.csv")
    skill_chart(pd.read_csv(final / "test_metrics.csv"))
    print("map hour:", snapshot_map(cfg))


if __name__ == "__main__":
    main()
