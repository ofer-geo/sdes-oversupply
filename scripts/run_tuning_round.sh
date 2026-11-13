#!/usr/bin/env bash
# Score the benchmarks and all neural variants on the tuning period. Finished steps are skipped,
# so the script can simply be run again after an interruption.
# Run from the repository root with the environment's Python as $PY.
set -euo pipefail
PROCESSED=data/processed
LOGS=$PROCESSED/neural/logs
mkdir -p "$LOGS"

[ -f "$PROCESSED/panel/cells.parquet" ] || "$PY" scripts/build_panel.py > "$LOGS/panel.log" 2>&1
[ -f "$PROCESSED/panel/flow_graph.npz" ] || "$PY" scripts/build_flow_graph.py > "$LOGS/flow.log" 2>&1
[ -f "$PROCESSED/benchmarks/benchmarks_tune.csv" ] || \
  "$PY" scripts/run_benchmarks.py > "$LOGS/benchmarks.log" 2>&1

for weight in 0 0.2; do
  for model in gru gnn gnn_flow; do
    if [ -f "$PROCESSED/neural/${model}_w${weight}_seed42_tune.csv" ]; then
      echo "$model weight $weight already done"
      continue
    fi
    "$PY" scripts/train_nn.py --model "$model" --busy-weight "$weight" \
      > "$LOGS/${model}_w${weight}.log" 2>&1
    echo "$model weight $weight done"
  done
done
