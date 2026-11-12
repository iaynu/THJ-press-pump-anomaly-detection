"""Data loading and gap-safe segmentation.

A 'segment' is a run of rows recorded continuously at ~0.1 s spacing. A new segment starts
where the time step is > 0.5 s or not increasing (dt <= 0). Exact duplicate rows are removed
first. Windows are only ever cut INSIDE a segment, and every train/test split is made on
segments BEFORE any window is cut, so overlapping windows never leak across a split.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

SENSORS = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
GAP_SEC = 0.5


@dataclass
class Segment:
    seg_id: int
    x: np.ndarray            # (rows, 3): upper vibration, lower vibration, current
    t0: pd.Timestamp         # first timestamp
    t1: pd.Timestamp         # last timestamp
    gap_before: float        # seconds since the previous segment ended (nan for the first)
    label: int               # Equipment_state of the file (0 normal, 1 abnormal)
    row_index: np.ndarray    # original row numbers in the file (for writing predictions back)

    @property
    def n(self):
        return len(self.x)


def load_table(path):
    d = pd.read_csv(path, index_col=0)
    d["TimeStamp"] = pd.to_datetime(d["TimeStamp"])
    d = d.drop_duplicates(subset=["TimeStamp"] + SENSORS).sort_values("TimeStamp", kind="stable")
    return d


def segment_table(d, label=None):
    t = d["TimeStamp"]
    dt = t.diff().dt.total_seconds()
    new = (dt > GAP_SEC) | (dt <= 0) | dt.isna()
    if "_break" in d:            # a dropped (invalid) row sits here: the sampling is not continuous across it
        new |= d["_break"].to_numpy(bool)
    gid = new.cumsum().to_numpy()
    segs, prev_end = [], None
    lab = d["Equipment_state"].to_numpy() if label is None else np.full(len(d), label)
    for k in np.unique(gid):
        m = gid == k
        tt = t[m]
        gap = np.nan if prev_end is None else (tt.iloc[0] - prev_end).total_seconds()
        segs.append(Segment(int(k), d.loc[m, SENSORS].to_numpy(float), tt.iloc[0], tt.iloc[-1], gap,
                            int(lab[m][0]), d.index[m].to_numpy()))
        prev_end = tt.iloc[-1]
    return segs


def load_segments(path, label=None):
    return segment_table(load_table(path), label)
