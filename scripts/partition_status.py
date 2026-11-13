"""Split the monthly Status Changes archives into vehicle partitions (data/interim).

    python scripts/partition_status.py [--config config/default.yaml]

Safe to re-run: finished months are skipped.
"""

import argparse
import time

from oversupply.config import DEFAULT_CONFIG, Config
from oversupply.io import partition_month


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = Config.load(args.config)
    out = cfg.paths.interim / "status_partitions"
    out.mkdir(parents=True, exist_ok=True)

    archives = sorted(cfg.paths.raw_status_changes.glob("*.rar"))
    if not archives:
        raise FileNotFoundError(f"No .rar archives in {cfg.paths.raw_status_changes}")
    for archive in archives:
        t0 = time.perf_counter()
        rows = partition_month(archive, out, cfg.partitioning.n_buckets)
        state = "already done" if rows is None else f"{rows:,} rows"
        print(f"{archive.stem}: {state} in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
