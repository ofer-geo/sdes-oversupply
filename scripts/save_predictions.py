"""Save forecasts of the trained LightGBM and neural models for one period.

    python scripts/save_predictions.py --split tune

Writes data/processed/predictions/<split>_<model>.npz: one cells x target-hours array per reported
horizon. The test split is only used for the final, once-only evaluation.
"""

import argparse
import re

import lightgbm as lgb
import numpy as np
import torch

from oversupply.config import DEFAULT_CONFIG, Config
from oversupply.data import load_model_data
from oversupply.models import Forecaster, ModelSize, TensorPanel, load_graphs, predict_rates

CHECKPOINT = re.compile(r"(gru|gnn_flow|gnn)_w[\d.]+_seed\d+")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=["tune", "test"], required=True)
    parser.add_argument("--neural", nargs="+", required=True,
                        help="checkpoint names, e.g. gnn_flow_w0_seed42")
    args = parser.parse_args()
    cfg = Config.load(args.config)
    data = load_model_data(cfg)
    panel = data.panel
    lo, hi = ((data.tune_lo, data.test_lo) if args.split == "tune"
              else (data.test_lo, data.test_hi))
    targets = np.arange(lo, hi)
    n_cells, n_targets = panel.parked.shape[0], len(targets)
    horizons = cfg.forecast.horizons
    H = cfg.forecast.max_horizon
    out = cfg.paths.processed / "predictions"
    out.mkdir(parents=True, exist_ok=True)

    # Finished files are skipped, so an interrupted run can simply be started again.
    light_path = out / f"{args.split}_LightGBM.npz"
    if not light_path.exists():
        booster = lgb.Booster(model_file=cfg.paths.processed / "benchmarks" / "lightgbm.txt")
        forecasts = {}
        for horizon in horizons:
            cell = np.repeat(np.arange(n_cells), n_targets)
            origin = np.tile(targets - horizon, n_cells)
            pred = np.empty(len(cell), dtype=np.float32)
            for start in range(0, len(cell), 500_000):
                part = slice(start, start + 500_000)
                pred[part] = booster.predict(panel.rows(cell[part], origin[part],
                                                        np.full(len(pred[part]), horizon)))
            forecasts[f"h{horizon}"] = pred.reshape(n_cells, n_targets)
        np.savez_compressed(light_path, **forecasts)
        print("LightGBM saved", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensors = TensorPanel(panel, H, device)
    origins = np.arange(lo - H, hi - 1)
    for name in args.neural:
        match = CHECKPOINT.fullmatch(name)
        if match is None:
            raise ValueError(f"Unrecognized checkpoint name: {name}")
        path = out / f"{args.split}_{name}.npz"
        if path.exists():
            continue
        graphs = load_graphs(match.group(1), data.adjacency,
                             cfg.paths.processed / "panel" / "flow_graph.npz", device)
        model = Forecaster(tensors.n_nodes, tensors.n_static, H, tensors.statics, graphs,
                           ModelSize()).to(device)
        model.load_state_dict(torch.load(cfg.paths.processed / "neural" / f"{name}.pt",
                                         weights_only=True))
        rates = predict_rates(model, tensors, origins)
        forecasts = {f"h{h}": rates[targets - h - origins[0], :, h - 1].T for h in horizons}
        np.savez_compressed(path, **forecasts)
        print(f"{name} saved", flush=True)


if __name__ == "__main__":
    main()
