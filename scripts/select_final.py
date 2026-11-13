"""Fix every remaining choice on the tuning period, before the test period is opened.

    python scripts/select_final.py --gnn gnn_flow_w0_seed42 gnn_flow_w0_seed43 gnn_flow_w0_seed44

Reports seed noise, blends LightGBM with the mean of the GNN seeds, chooses flagging cutoffs, and
measures how far the models agree. Choices are saved to data/processed/final/choices.json.
"""

import argparse
import json

import numpy as np

from oversupply.config import DEFAULT_CONFIG, Config
from oversupply.data import load_model_data
from oversupply.metrics import (
    K_BY_LEVEL,
    average_precision,
    best_cutoff,
    blend,
    evaluate,
    top_flags,
)

SELECTION_HORIZONS = (3, 6)
WEIGHTS = np.round(np.arange(0.0, 1.01, 0.1), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--gnn", nargs="+", required=True)
    args = parser.parse_args()
    cfg = Config.load(args.config)
    data = load_model_data(cfg)
    p = data.panel.parked
    targets = np.arange(data.tune_lo, data.test_lo)
    y = p[:, targets]
    folder = cfg.paths.processed / "predictions"
    horizons = cfg.forecast.horizons
    levels = tuple(cfg.forecast.levels)

    light = np.load(folder / "tune_LightGBM.npz")
    seeds = {name: np.load(folder / f"tune_{name}.npz") for name in args.gnn}
    ap20 = lambda pred: average_precision(pred.ravel(), (y >= 20).ravel())  # noqa: E731

    print("Seed noise, AP 20+:")
    for h in horizons:
        scores = [ap20(seed[f"h{h}"]) for seed in seeds.values()]
        print(f"  {h} h: " + ", ".join(f"{s:.3f}" for s in scores)
              + f" | mean {np.mean(scores):.3f}, sd {np.std(scores, ddof=1):.3f}", flush=True)
    gnn = {h: np.mean([seed[f"h{h}"] for seed in seeds.values()], axis=0) for h in horizons}

    print("\nBlend weight on LightGBM (mean AP 20+ at 3 and 6 h):")
    curve = {}
    for w in WEIGHTS:
        curve[float(w)] = np.mean([ap20(blend(light[f"h{h}"], gnn[h], w))
                                   for h in SELECTION_HORIZONS])
        print(f"  {w:.1f}: {curve[float(w)]:.4f}", flush=True)
    weight = max(curve, key=curve.get)
    print(f"chosen weight {weight}")

    models = {"LightGBM": lambda h: light[f"h{h}"], "GNN + flows (mean of seeds)": lambda h: gnn[h],
              "Blend": lambda h: blend(light[f"h{h}"], gnn[h], weight)}
    cutoffs = {}
    print()
    for name, forecast in models.items():
        cutoffs[name] = {}
        for h in horizons:
            pred = forecast(h)
            metrics = evaluate(y, pred, p[:, targets - h], levels, K_BY_LEVEL)
            cutoffs[name][str(h)] = {}
            lines = []
            for level in levels:
                positive = y >= level
                candidates = np.linspace(0.3 * level, 1.1 * level, 33)
                cutoff, f1 = best_cutoff(pred, positive, candidates)
                cutoffs[name][str(h)][str(level)] = cutoff
                flagged = pred >= cutoff
                hits = (flagged & positive).sum()
                lines.append(f"{level}+: AP {metrics[f'ap_{level}']:.3f}, cutoff {cutoff:.1f}, "
                             f"precision {hits / max(flagged.sum(), 1):.3f}, "
                             f"recall {hits / max(positive.sum(), 1):.3f}, F1 {f1:.3f}")
            k = K_BY_LEVEL[20]
            print(f"{name}, {h} h | MAE {metrics['mae']:.3f} | " + " | ".join(lines)
                  + f" | new 20+: precision {metrics[f'new_precision_top{k}_20']:.3f}, "
                  f"recall {metrics[f'new_recall_top{k}_20']:.3f}", flush=True)

    print("\nDo the two models find the same new 20+ pile-ups? (10 flags per hour)")
    for h in horizons:
        below = p[:, targets - h] < 20
        new = (y >= 20) & below
        a = top_flags(light[f"h{h}"], below, K_BY_LEVEL[20]) & new
        b = top_flags(gnn[h], below, K_BY_LEVEL[20]) & new
        total = max(int(new.sum()), 1)
        print(f"  {h} h: new {total:,}; both {100 * (a & b).sum() / total:.1f}%, "
              f"LightGBM only {100 * (a & ~b).sum() / total:.1f}%, "
              f"GNN only {100 * (b & ~a).sum() / total:.1f}%, "
              f"neither {100 * (new & ~a & ~b).sum() / total:.1f}%; forecast correlation "
              f"{np.corrcoef(light[f'h{h}'].ravel(), gnn[h].ravel())[0, 1]:.3f}", flush=True)

    out = cfg.paths.processed / "final"
    out.mkdir(parents=True, exist_ok=True)
    (out / "choices.json").write_text(json.dumps({
        "gnn_checkpoints": args.gnn, "blend_weight_lightgbm": weight, "cutoffs": cutoffs,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
