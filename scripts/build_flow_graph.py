"""Build the scooter-flow graph between modelled cells, from training months only.

    python scripts/build_flow_graph.py

Each vehicle partition is read separately, because it holds that vehicle's complete history.
"""

import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
import shapely

from oversupply.config import DEFAULT_CONFIG, Config
from oversupply.data import load_model_data
from oversupply.flows import flow_adjacency, vehicle_moves
from oversupply.panel import cell_key

MAX_GAP_H = 2.0
TOP_SOURCES = 20
MIN_RIDES = 5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = Config.load(args.config)
    keys = load_model_data(cfg).keys
    cutoff = pd.Timestamp(cfg.splits.tune_start)

    moves = []
    for part in sorted((cfg.paths.interim / "event_parts").glob("part-*.parquet")):
        table = pq.read_table(part, columns=["device_id", "start_time", "end_time", "end_reason",
                                             "geometry"])
        points = shapely.from_wkb(table["geometry"].to_numpy())
        events = table.drop(["geometry"]).to_pandas()
        events["key"] = cell_key(shapely.get_x(points), shapely.get_y(points),
                                 cfg.panel.cell_size_m)
        # Only rides completed before the tuning period may shape the graph
        events = events[events["end_time"] < cutoff]
        moves.append(vehicle_moves(events, MAX_GAP_H))
    moves = pd.concat(moves, ignore_index=True)
    graph = flow_adjacency(moves, keys, TOP_SOURCES, MIN_RIDES)
    out = cfg.paths.processed / "panel" / "flow_graph.npz"
    sp.save_npz(out, graph)
    links = np.asarray((graph > 0).sum(axis=1)).ravel()
    print(f"{len(moves):,} moves; {graph.nnz:,} links; cells with incoming links "
          f"{(links > 0).sum():,}/{len(keys):,}; mean links {links.mean():.1f}")


if __name__ == "__main__":
    main()
