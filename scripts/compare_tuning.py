"""Rank all models on the tuning period: mean average precision at 20+, across 3 and 6 hours.

    python scripts/compare_tuning.py
"""

import pandas as pd

from oversupply.config import Config

COLUMNS = {
    "ap_10": "AP 10+", "ap_20": "AP 20+", "ap_40": "AP 40+", "mae": "MAE",
    "new_recall_top10_20": "new 20+ recall", "new_precision_top10_20": "new 20+ precision",
    "precision_20": "precision 20+", "recall_20": "recall 20+",
}


def main() -> None:
    cfg = Config.load()
    frames = [pd.read_csv(cfg.paths.processed / "benchmarks" / "benchmarks_tune.csv")]
    for path in sorted((cfg.paths.processed / "neural").glob("*_tune.csv")):
        frames.append(pd.read_csv(path).assign(
            model=lambda d, name=path.stem: name.removesuffix("_tune")))
    df = pd.concat(frames, ignore_index=True)
    table = df[["horizon_h", "model", *COLUMNS]].rename(columns=COLUMNS).round(3)
    for horizon, rows in table.groupby("horizon_h"):
        print(f"\n{horizon} h ahead\n" + rows.drop(columns="horizon_h")
              .sort_values("AP 20+", ascending=False).to_string(index=False))
    score = df[df["horizon_h"].isin([3, 6])].groupby("model")["ap_20"].mean()
    print("\nSelection score (mean AP 20+ at 3 and 6 h)\n"
          + score.sort_values(ascending=False).round(4).to_string())


if __name__ == "__main__":
    main()
