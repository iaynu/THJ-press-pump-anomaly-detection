import numpy as np
import pandas as pd

from kamp.data import segment_table, load_segments


def _table(times):
    n = len(times)
    return pd.DataFrame({"TimeStamp": pd.to_datetime(times), "AI0_Vibration": np.arange(n, dtype=float),
                         "AI1_Vibration": np.zeros(n), "AI2_Current": np.ones(n)})


def test_gap_over_half_second_starts_new_segment():
    t0 = pd.Timestamp("2024-01-01 00:00:00")
    times = [t0 + pd.Timedelta(milliseconds=100 * i) for i in range(10)]
    times += [t0 + pd.Timedelta(seconds=5, milliseconds=100 * i) for i in range(10)]
    segs = segment_table(_table(times), label=0)
    assert [s.n for s in segs] == [10, 10]
    assert segs[1].gap_before > 0.5


def test_continuous_rows_are_one_segment():
    t0 = pd.Timestamp("2024-01-01")
    segs = segment_table(_table([t0 + pd.Timedelta(milliseconds=100 * i) for i in range(30)]), label=0)
    assert len(segs) == 1 and segs[0].n == 30


def test_shipped_data_segment_counts():
    """README: 599 normal segments, 21 abnormal segments"""
    assert len(load_segments("data/press_data_normal.csv", 0)) == 599
    assert len(load_segments("data/outlier_data.csv", 1)) == 21
