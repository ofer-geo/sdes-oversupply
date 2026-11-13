"""Benchmarks on the tuning period (Sep-Oct). The test period stays unseen.

    python scripts/run_benchmarks.py

Compares staying the same, same hour last week, the cell's usual level, and LightGBM.
"""

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from oversupply.config import DEFAULT_CONFIG, Config
from oversupply.data import load_model_data
from oversupply.features import Panel
from oversupply.metrics import K_BY_LEVEL, evaluate

SEED = 42
N_CANDIDATES = 6_000_000
QUIET_KEEP = 0.25          # share of low-count training rows kept; they are reweighted to match


def sample_rows(panel: Panel, lo: int, hi: int, max_h: int, n: int, rng) -> tuple:
    """Random forecasts whose target falls in [lo, hi). Quiet rows are thinned and reweighted."""
    cell = rng.integers(0, panel.parked.shape[0], n)
    h = rng.integers(1, max_h + 1, n)
    target = rng.integers(lo, hi, n)
    y = panel.parked[cell, target]
    keep = (y >= 10) | (rng.random(n) < QUIET_KEEP)
    weight = np.where(y >= 10, 1.0, 1.0 / QUIET_KEEP)[keep]
    return cell[keep], target[keep] - h[keep], h[keep], y[keep], weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = Config.load(args.config)
    t_start = time.perf_counter()
    data = load_model_data(cfg)
    panel = data.panel
    rng = np.random.default_rng(SEED)
    max_h = cfg.forecast.max_horizon

    cell, t, h, y, weight = sample_rows(panel, data.train_lo, data.tune_lo, max_h, N_CANDIDATES,
                                        rng)
    train = lgb.Dataset(panel.rows(cell, t, h), y, weight=weight, free_raw_data=True)
    print(f"training rows {len(y):,} ({time.perf_counter() - t_start:.0f}s)", flush=True)
    cell, t, h, y, weight = sample_rows(panel, data.tune_lo, data.test_lo, max_h,
                                        N_CANDIDATES // 6, rng)
    valid = lgb.Dataset(panel.rows(cell, t, h), y, weight=weight, reference=train)
    model = lgb.train(
        {"objective": "poisson", "metric": "l1", "learning_rate": 0.05, "num_leaves": 127,
         "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
         "bagging_freq": 1, "lambda_l2": 1.0, "num_threads": 6, "seed": SEED, "verbosity": -1},
        train, num_boost_round=800, valid_sets=[valid],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(100)])
    print(f"trees {model.best_iteration} ({time.perf_counter() - t_start:.0f}s)", flush=True)

    out_dir = cfg.paths.processed / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(out_dir / "lightgbm.txt", num_iteration=model.best_iteration)
    importance = pd.Series(model.feature_importance("gain"), index=model.feature_name())
    print((importance / importance.sum() * 100).sort_values(ascending=False).head(15).round(1)
          .to_string(), flush=True)

    # The evaluation covers the modelled cells: those that reached min_active_peak in training.
    p = panel.parked
    targets = np.arange(data.tune_lo, data.test_lo)
    n_cells, n_targets = p.shape[0], len(targets)
    results = []
    for horizon in cfg.forecast.horizons:
        y_true = p[:, targets]
        origin = p[:, targets - horizon]
        cell = np.repeat(np.arange(n_cells), n_targets)
        o = np.tile(targets - horizon, n_cells)
        hs = np.full(len(cell), horizon)
        pred = np.empty(len(cell), dtype=np.float32)
        for start in range(0, len(cell), 2_000_000):
            part = slice(start, start + 2_000_000)
            pred[part] = model.predict(panel.rows(cell[part], o[part], hs[part]),
                                       num_iteration=model.best_iteration)
        forecasts = {
            "stays the same": origin.astype(np.float32),
            "same hour last week": p[:, targets - 168].astype(np.float32),
            "usual level (cell, weekday, hour)": panel.profile[:, panel.slot[targets]],
            "LightGBM": pred.reshape(n_cells, n_targets),
        }
        for name, f in forecasts.items():
            row = {"horizon_h": horizon, "model": name,
                   **evaluate(y_true, f, origin, tuple(cfg.forecast.levels), K_BY_LEVEL)}
            results.append(row)
            print(json.dumps({k: round(v, 3) if isinstance(v, float) else v
                              for k, v in row.items()}), flush=True)
    pd.DataFrame(results).to_csv(out_dir / "benchmarks_tune.csv", index=False)
    print(f"done in {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
