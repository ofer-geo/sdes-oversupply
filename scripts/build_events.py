"""Build the parking-events dataset, one vehicle partition at a time.

    python scripts/build_events.py [--config config/default.yaml]

Each partition's events and funnel are saved as soon as it's done, so a re-run resumes after the
last finished partition. The funnel CSV is written last and marks a partition as complete.
"""

import argparse
import time

import pandas as pd

from oversupply.config import DEFAULT_CONFIG, Config
from oversupply.events import ParkingEventBuilder
from oversupply.geo import Layers
from oversupply.io import merge_geoparquet, read_bucket, write_geoparquet
from oversupply.report import FilterReport


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = Config.load(args.config)
    partitions = cfg.paths.interim / "status_partitions"
    parts_dir = cfg.paths.interim / "event_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.processed.mkdir(parents=True, exist_ok=True)
    layers = Layers(cfg)
    t0 = time.perf_counter()

    for bucket in range(cfg.partitioning.n_buckets):
        part = parts_dir / f"part-{bucket:02d}.parquet"
        funnel = part.with_suffix(".csv")
        if funnel.exists():
            continue
        builder = ParkingEventBuilder(cfg, layers)
        events = builder.run(read_bucket(partitions, bucket))
        write_geoparquet(events, part)
        builder.report.to_frame().to_csv(funnel, index=False)
        print(f"partition {bucket:02d}: {len(events):,} events "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

    parts = [parts_dir / f"part-{b:02d}.parquet" for b in range(cfg.partitioning.n_buckets)]
    total = merge_geoparquet(parts, cfg.paths.processed / "parking_events.parquet")
    report = FilterReport.from_frames("Parking events", [
        pd.read_csv(p.with_suffix(".csv")) for p in parts])
    report.to_frame().to_csv(cfg.paths.processed / "parking_events_report.csv", index=False)
    print(report)
    print(f"\n{total:,} events; {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
