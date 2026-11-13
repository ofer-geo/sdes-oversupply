"""Train a neural forecaster and score it on the tuning period (Sep-Oct).

    python scripts/train_nn.py --model gru
    python scripts/train_nn.py --model gnn                  # touching cells
    python scripts/train_nn.py --model gnn_flow             # touching cells + scooter flows
    python scripts/train_nn.py --model gnn --busy-weight 0.2

The test period is never touched here.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from oversupply.config import DEFAULT_CONFIG, Config
from oversupply.data import load_model_data
from oversupply.metrics import K_BY_LEVEL, evaluate
from oversupply.models import (
    Forecaster,
    ModelSize,
    TensorPanel,
    busy_weights,
    load_graphs,
    predict_rates,
)

BATCH_SIZE = 8         # origins per batch; each holds every modelled cell
PREDICT_BATCH = 16
LEARNING_RATE = 1e-3
EPOCHS = 12
PATIENCE = 3


def weighted_loss(log_rate: torch.Tensor, y: torch.Tensor, strength: float) -> torch.Tensor:
    loss = F.poisson_nll_loss(log_rate, y, log_input=True, reduction="none")
    return (loss * busy_weights(y, strength)).mean()


@torch.no_grad()
def tune_loss(model, tensors, origins: np.ndarray, strength: float) -> float:
    model.eval()
    total, batches = 0.0, 0
    for start in range(0, len(origins), PREDICT_BATCH):
        chunk = torch.as_tensor(origins[start:start + PREDICT_BATCH], device=tensors.device)
        total += weighted_loss(model(tensors.batch(chunk)), tensors.targets(chunk),
                               strength).item()
        batches += 1
    return total / batches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model", choices=["gru", "gnn", "gnn_flow"], required=True)
    parser.add_argument("--busy-weight", type=float, default=0.0,
                        help="extra loss weight per parked scooter in the target")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    cfg = Config.load(args.config)
    t_start = time.perf_counter()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}", flush=True)

    data = load_model_data(cfg)
    H = cfg.forecast.max_horizon
    tensors = TensorPanel(data.panel, H, device)
    graphs = load_graphs(args.model, data.adjacency,
                         cfg.paths.processed / "panel" / "flow_graph.npz", device)
    model = Forecaster(tensors.n_nodes, tensors.n_static, H, tensors.statics, graphs,
                       ModelSize()).to(device)
    n_parameters = sum(p.numel() for p in model.parameters())
    name = f"{args.model}_w{args.busy_weight:g}_seed{args.seed}"
    print(f"{name}: {n_parameters:,} parameters, {tensors.n_nodes:,} cells", flush=True)

    # An origin is usable if all of its targets fall inside the split
    train_origins = np.arange(data.train_lo - 1, data.tune_lo - H)
    tune_origins = np.arange(data.tune_lo - 1, data.test_lo - H)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    steps = args.epochs * (len(train_origins) // BATCH_SIZE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, LEARNING_RATE, total_steps=steps)

    out_dir = cfg.paths.processed / "neural"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / f"{name}.pt"
    best, stale = float("inf"), 0
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(train_origins)
        running, batches = 0.0, 0
        for start in range(0, len(order) - BATCH_SIZE + 1, BATCH_SIZE):
            chunk = torch.as_tensor(order[start:start + BATCH_SIZE], device=device)
            loss = weighted_loss(model(tensors.batch(chunk)), tensors.targets(chunk),
                                 args.busy_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
            batches += 1
        loss_tune = tune_loss(model, tensors, tune_origins[::3], args.busy_weight)
        print(f"epoch {epoch + 1}: train {running / batches:.4f}, tune {loss_tune:.4f} "
              f"({time.perf_counter() - t_start:.0f}s)", flush=True)
        if loss_tune < best - 1e-4:
            best, stale = loss_tune, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= PATIENCE:
                print("early stop", flush=True)
                break

    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    # Every tuning hour must be reachable at every horizon. The last origins also forecast a few
    # hours beyond the split; those outputs are discarded and never scored.
    eval_origins = np.arange(data.tune_lo - H, data.test_lo - 1)
    rates = predict_rates(model, tensors, eval_origins, PREDICT_BATCH)
    p = data.panel.parked
    targets = np.arange(data.tune_lo, data.test_lo)
    results = []
    for horizon in cfg.forecast.horizons:
        # Target hour tau comes from the origin tau - horizon
        pick = targets - horizon - eval_origins[0]
        pred = rates[pick, :, horizon - 1].T
        row = {"horizon_h": horizon, "model": args.model, "busy_weight": args.busy_weight,
               "seed": args.seed,
               **evaluate(p[:, targets], pred, p[:, targets - horizon],
                          tuple(cfg.forecast.levels), K_BY_LEVEL)}
        results.append(row)
        print(json.dumps({k: round(v, 3) if isinstance(v, float) else v
                          for k, v in row.items()}), flush=True)
    pd.DataFrame(results).to_csv(out_dir / f"{name}_tune.csv", index=False)
    print(f"parameters {n_parameters:,}; done in {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
