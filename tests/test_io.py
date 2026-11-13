import pandas as pd

from oversupply.io import bucket_of, partition_chunks, read_bucket
from oversupply.report import FilterReport


def chunk(month: str, device_ids: list[str]) -> pd.DataFrame:
    n = len(device_ids)
    return pd.DataFrame({
        "device_id": pd.array(device_ids, dtype="string"),
        "vehicle_state": pd.array(["available"] * n, dtype="string"),
        "event_types": pd.array(["trip_end"] * n, dtype="string"),
        "event_time": pd.date_range(f"{month}-01", periods=n, freq="h").astype(str),
        "vehicle_type": pd.array(["scooter"] * n, dtype="string"),
        "lat": [32.0] * n,
        "lon": [34.0] * n,
    })


def test_vehicle_history_stays_in_one_bucket_across_months(tmp_path):
    ids = [f"vehicle-{i}" for i in range(50)]
    for month in ("2024-01", "2024-02"):
        partition_chunks([chunk(month, ids)], tmp_path / month, n_buckets=4)

    buckets = bucket_of(pd.Series(ids, dtype="string"), 4)
    assert len(set(buckets)) > 1
    for b in range(4):
        df = read_bucket(tmp_path, b)
        expected = {i for i, n in zip(ids, buckets, strict=True) if n == b}
        assert set(df["device_id"]) == expected
        assert len(df) == 2 * len(expected)
        # January comes before February, as event building requires
        for _, rows in df.groupby("device_id", observed=True):
            assert rows["event_time"].is_monotonic_increasing


def test_chunks_keep_row_order_within_a_vehicle(tmp_path):
    first = chunk("2024-01", ["same"] * 3)
    second = chunk("2024-01", ["same"] * 3).assign(
        event_time=pd.date_range("2024-01-02", periods=3, freq="h").astype(str))
    partition_chunks([first, second], tmp_path / "2024-01", n_buckets=2)
    bucket = int(bucket_of(pd.Series(["same"], dtype="string"), 2).iloc[0])
    assert read_bucket(tmp_path, bucket)["event_time"].is_monotonic_increasing


def test_funnels_of_partitions_add_up():
    a, b = FilterReport("x"), FilterReport("x")
    a.start(10)
    a.add("step", 3)
    b.start(5)
    b.add("step", 1)
    total = FilterReport.from_frames("x", [a.to_frame(), b.to_frame()])
    assert (total.initial, total.remaining) == (15, 11)
