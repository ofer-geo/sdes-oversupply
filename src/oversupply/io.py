"""Stream monthly Status Changes archives into vehicle partitions.

A parking event can span a month boundary, and building events needs a vehicle's records in
time order. A full year doesn't fit in memory on a laptop, so each month is split by a hash of
``device_id``: one bucket holds the complete history of its vehicles and is small enough to
process alone. Archives are decompressed to a pipe, never unpacked to disk.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

STATUS_COLUMNS = ["device_id", "vehicle_state", "event_types", "event_time", "vehicle_type",
                  "lat", "lon"]
_STATUS_SCHEMA = pa.schema([
    ("device_id", pa.string()),
    ("vehicle_state", pa.string()),
    ("event_types", pa.string()),
    ("event_time", pa.timestamp("us")),
    ("vehicle_type", pa.string()),
    ("lat", pa.float64()),
    ("lon", pa.float64()),
])
ROW_GROUP_SIZE = 20_000


def _bsdtar() -> str:
    candidates = [Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "tar.exe"]
    if sys.platform != "win32":
        candidates = [Path(p) for p in (shutil.which("bsdtar"), shutil.which("tar")) if p]
    for c in candidates:
        if c.exists():
            return str(c)
    raise RuntimeError("bsdtar is required to read .rar archives")


@contextlib.contextmanager
def open_rar(path: str | Path):
    """Open the single CSV inside a .rar archive as a stream."""
    proc = subprocess.Popen([_bsdtar(), "-xOf", str(path)], stdout=subprocess.PIPE,
                            bufsize=1 << 22)
    try:
        yield proc.stdout
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def iter_status_chunks(archive: str | Path, chunksize: int = 200_000) -> Iterator[pd.DataFrame]:
    with open_rar(archive) as f:
        yield from pd.read_csv(f, usecols=STATUS_COLUMNS, chunksize=chunksize,
                               dtype={"device_id": "string", "vehicle_state": "string",
                                      "event_types": "string", "vehicle_type": "string"})


def bucket_of(device_id: pd.Series, n_buckets: int) -> pd.Series:
    """Stable partition number of each vehicle (the same across months and runs)."""
    hashed = pd.util.hash_pandas_object(device_id, index=False).to_numpy()
    return pd.Series(hashed % n_buckets, index=device_id.index)


def partition_chunks(chunks: Iterable[pd.DataFrame], out_dir: str | Path, n_buckets: int) -> int:
    """Write chunks to ``out_dir/bucket-NN.parquet``, keeping each vehicle's record order."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[int, pq.ParquetWriter] = {}
    total = 0
    try:
        for chunk in chunks:
            chunk = chunk.assign(event_time=pd.to_datetime(chunk["event_time"]))
            total += len(chunk)
            for bucket, rows in chunk.groupby(bucket_of(chunk["device_id"], n_buckets),
                                              sort=False):
                bucket = int(bucket)
                if bucket not in writers:
                    writers[bucket] = pq.ParquetWriter(out_dir / f"bucket-{bucket:02d}.parquet",
                                                       _STATUS_SCHEMA, compression="zstd")
                writers[bucket].write_table(pa.Table.from_pandas(
                    rows[_STATUS_SCHEMA.names], schema=_STATUS_SCHEMA, preserve_index=False))
    finally:
        for writer in writers.values():
            writer.close()
    return total


def partition_month(archive: str | Path, partitions_dir: str | Path, n_buckets: int) -> int | None:
    """Partition one archive into ``partitions_dir/<month>/``; a finished month is skipped.

    Returns the number of rows, or None if the month was already done. Files go to a temporary
    folder that is renamed at the end, so an interrupted run never leaves a partial month.
    """
    archive = Path(archive)
    final = Path(partitions_dir) / archive.stem
    if final.exists():
        return None
    tmp = Path(partitions_dir) / f".tmp-{archive.stem}"
    shutil.rmtree(tmp, ignore_errors=True)
    total = partition_chunks(iter_status_chunks(archive), tmp, n_buckets)
    tmp.rename(final)
    return total


def read_bucket(partitions_dir: str | Path, bucket: int) -> pd.DataFrame:
    """All records of one vehicle partition, months in chronological order."""
    name = f"bucket-{bucket:02d}.parquet"
    files = [p / name for p in sorted(Path(partitions_dir).glob("[0-9]*")) if (p / name).exists()]
    if not files:
        return pd.DataFrame(columns=_STATUS_SCHEMA.names)
    df = pa.concat_tables([pq.read_table(f) for f in files]).to_pandas()
    return df.astype({"device_id": "string", "vehicle_state": "category",
                      "event_types": "category", "vehicle_type": "category"})


def write_geoparquet(gdf, path: str | Path) -> None:
    gdf.to_parquet(path, compression="zstd", row_group_size=ROW_GROUP_SIZE)


def merge_geoparquet(parts: list[Path], out_path: str | Path) -> int:
    """Stream GeoParquet parts into one file. The first part's bbox would be wrong, so it's cut."""
    first = pq.ParquetFile(parts[0]).schema_arrow
    geo = orjson.loads(first.metadata[b"geo"])
    for column in geo["columns"].values():
        column.pop("bbox", None)
    schema = first.with_metadata({**first.metadata, b"geo": orjson.dumps(geo)})
    total = 0
    with pq.ParquetWriter(out_path, schema, compression="zstd") as writer:
        for part in parts:
            table = pq.read_table(part).cast(schema)
            writer.write_table(table, row_group_size=ROW_GROUP_SIZE)
            total += table.num_rows
    return total
