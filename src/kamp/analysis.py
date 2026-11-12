"""File-level analysis shared by `run.py predict` and the dashboard's file-analysis page.

    raw CSV rows -> validation -> clean rows -> segments -> fused normal percentile per segment
    -> level (normal / attention / warning) -> one result row for EVERY input row

Every input row stays in the output. Rows that are not scored themselves are kept and marked:
  row_status = "분석"      scored as part of its segment
             = "중복 행"    exact duplicate of an earlier row (same time and sensor values); takes its twin's result
             = "제외"       missing / non-numeric sensor value or unreadable time; no result
Segments shorter than 8 rows cannot be scored; they take the decision of the nearest scored segment in time and are
flagged judged_directly = False.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .data import SENSORS, segment_table
from .features import FEATURES, CUR
from .system import fill_short, diagnose
from .evaluate import _ewma
from .detectors import clear_cache

LEVEL = {0: "정상", 1: "주의", 2: "경고"}
MIN_ROWS = 8
REQUIRED = ["TimeStamp"] + SENSORS


class InputError(ValueError):
    """the file cannot be analysed; the message is shown to the user as is"""


@dataclass
class Result:
    rows: pd.DataFrame                  # one row per input row, original index
    segments: pd.DataFrame              # one row per segment
    thresholds: dict
    counts: dict = field(default_factory=dict)


def read_csv(src):
    """path or file-like -> DataFrame with the original row index (first column if it is an index column)"""
    try:
        d = pd.read_csv(src)
    except Exception as e:                                                   # noqa: BLE001
        raise InputError(f"CSV 파일을 읽을 수 없습니다: {e}") from e
    if d.empty:
        raise InputError("파일에 데이터 행이 없습니다.")
    first = d.columns[0]
    if first in ("", "index") or str(first).startswith("Unnamed"):
        d = d.set_index(first)
        d.index.name = "index"
    else:
        d.index.name = "index"
    if not d.index.is_unique:
        d = d.reset_index(drop=True)
        d.index.name = "index"
    return d


def validate(d):
    """returns (clean rows, report). Raises InputError if nothing can be analysed."""
    missing = [c for c in REQUIRED if c not in d.columns]
    if missing:
        raise InputError("필수 열이 없습니다: " + ", ".join(missing) + ". 필요한 열: " + ", ".join(REQUIRED))
    x = d.copy()
    x["TimeStamp"] = pd.to_datetime(x["TimeStamp"], errors="coerce")
    for c in SENSORS:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    bad = x[REQUIRED].isna().any(axis=1)
    ok = x[~bad].copy()
    # a row dropped for a bad value leaves a hole in the 0.1 s sampling: start a new segment after it
    ok["_break"] = bad.shift(1, fill_value=False).cumsum().loc[ok.index].diff().fillna(0).gt(0)
    dup = ok.duplicated(subset=REQUIRED, keep="first")
    clean = ok[~dup].sort_values("TimeStamp", kind="stable")
    rep = {"rows_total": len(d), "rows_invalid": int(bad.sum()), "rows_duplicate": int(dup.sum()),
           "rows_clean": len(clean), "invalid_index": list(x.index[bad][:20]), "duplicate_index": list(ok.index[dup][:20])}
    if len(clean) < MIN_ROWS:
        raise InputError(f"분석 가능한 행이 {len(clean)}개뿐입니다. 최소 {MIN_ROWS}행(0.1초 간격 연속 측정)이 필요합니다."
                         + (f" 결측·숫자 아님·시각 오류로 제외된 행: {rep['rows_invalid']}개." if rep["rows_invalid"] else ""))
    return clean, rep


def _contrib_shares(det, x):
    c = det.window_contributions(x)
    if c is None:
        return None
    c = np.clip(c.mean(0), 0, None)
    return c / c.sum() if c.sum() > 0 else c


def analyze(d, B):
    """d: raw table from read_csv; B: model bundle (models/final_system.pkl)"""
    clean, rep = validate(d)
    S, tr = B["system"], B["trend"]
    clean = clean.copy()
    if "Equipment_state" not in clean:
        clean["Equipment_state"] = -1
    clean["Equipment_state"] = pd.to_numeric(clean["Equipment_state"], errors="coerce").fillna(-1).astype(int)
    segs = segment_table(clean)
    xs, ts = [s.x for s in segs], [s.t0 for s in segs]
    VP = S.view_p(xs)
    D2 = {n: S.det[n].seg_scores(xs) for n in ("joint", "current")}   # raw mean squared robust Mahalanobis distance
    lv, sc = S.levels(xs, VP)
    judged = ~np.isnan(lv)
    if not judged.any():
        raise InputError(f"모든 연속 구간이 {MIN_ROWS}행보다 짧아 판정할 수 없습니다. 0.5초 넘게 끊기지 않은 구간이 {MIN_ROWS}행 이상 있어야 합니다.")
    lv_f, sc_f = fill_short(lv, ts), fill_short(sc, ts)
    order = np.argsort(ts)
    z = S.z(sc_f[order])
    trend = np.zeros(len(segs), bool)
    trend[order] = (z > tr["strict_z"]) | (_ewma(z, tr.get("lambda", tr.get("lam", 0.2))) > tr["ewma_z"])

    seg_rows, row_rows = [], []
    for i, s in enumerate(segs):
        level = int(lv_f[i])
        diag, reason, top = "", "", ""
        if level >= 1:
            if judged[i]:
                diag = diagnose(B["vib_diag"], B["cur_diag"], s.x)
                view = "전류 전용" if VP[i, 1] > VP[i, 0] else "통합(진동+전류)"
                sh = _contrib_shares(S.det["joint"], s.x)
                if sh is not None:
                    k = np.argsort(sh)[::-1][:2]
                    top = ", ".join(f"{FEATURES[j]} {sh[j]:.0%}" for j in k)
                rar = 1 - max(VP[i])
                floor = 1 / (max(len(r) for r in S.ref.values()) + 1)
                reason = (f"{view} 관점: 정상 기준 조각 모두보다 드묾" if rar <= floor * 1.0001
                          else f"{view} 관점: 정상 대비 상위 {100 * rar:.2f}%")
            else:
                diag = "이웃 조각 판정 따름"
                reason = f"{len(s.x)}행이라 직접 판정 불가: 시간상 가장 가까운 조각의 판정을 따름"
        seg_rows.append({"segment_id": s.seg_id, "start_index": s.row_index[0], "end_index": s.row_index[-1],
                         "start_time": s.t0, "end_time": s.t1, "rows": s.n, "gap_before_s": s.gap_before,
                         "judged_directly": bool(judged[i]), "p_joint": VP[i, 0], "p_current": VP[i, 1],
                         "d2_joint": D2["joint"][i], "d2_current": D2["current"][i], "normal_percentile": sc_f[i], "rarity": 1 - sc_f[i], "level": level, "level_name": LEVEL[level],
                         "trend_warning": bool(trend[i]), "diagnosis": diag, "reason": reason, "top_features": top})
        for r in s.row_index:
            row_rows.append((r, s.seg_id))
    seg = pd.DataFrame(seg_rows)
    m = pd.DataFrame(row_rows, columns=["index", "segment_id"]).set_index("index")
    m = m.join(seg.set_index("segment_id")[["rows", "judged_directly", "p_joint", "p_current", "normal_percentile", "rarity",
                                            "level", "level_name", "trend_warning", "diagnosis"]], on="segment_id")
    m = m.rename(columns={"rows": "segment_rows"})

    # every input row comes back: duplicates take their twin's result, invalid rows are kept without a result
    out = d.copy()
    out = out.drop(columns=[c for c in ("_break",) if c in out])
    out["row_status"] = "제외"
    out = out.join(m)
    out.loc[m.index, "row_status"] = "분석"
    if rep["rows_duplicate"]:
        x = d.copy()
        x["TimeStamp"] = pd.to_datetime(x["TimeStamp"], errors="coerce")
        for c in SENSORS:
            x[c] = pd.to_numeric(x[c], errors="coerce")
        x = x.dropna(subset=REQUIRED)
        first = x.groupby(REQUIRED, sort=False).head(1)
        key = x[REQUIRED].astype(str).agg("|".join, axis=1)
        first_of = pd.Series(first.index, index=key.loc[first.index].values)
        dups = x.index[x.duplicated(subset=REQUIRED, keep="first")]
        twin = first_of.loc[key.loc[dups].values].values
        cols = list(m.columns)
        out.loc[dups, cols] = out.loc[twin, cols].values
        out.loc[dups, "row_status"] = "중복 행"
    out["pred_label"] = (out["level"] == 2).astype("Int64").where(out["row_status"] != "제외")
    out["level"] = out["level"].astype("Int64")
    out["segment_id"] = out["segment_id"].astype("Int64")
    clear_cache()          # feature cache is keyed by array id; do not let it grow across files

    n_seg = len(seg)
    counts = {**rep, "segments": n_seg, "segments_judged": int(judged.sum()), "segments_short": int((~judged).sum()),
              "seg_normal": int((seg.level == 0).sum()), "seg_attention": int((seg.level == 1).sum()),
              "seg_warning": int((seg.level == 2).sum()), "seg_trend": int(seg.trend_warning.sum()),
              "rows_analyzed": int((out.row_status == "분석").sum()),
              "rows_normal": int((out.level == 0).sum()), "rows_attention": int((out.level == 1).sum()),
              "rows_warning": int((out.level == 2).sum())}
    th = {"attention": float(S.t_att), "warning": float(S.t_warn),
          # smallest reportable rarity: the segment is beyond every normal reference segment
          "d2_joint_q99": float(np.quantile(S.ref["joint"], 0.99)),
          "rarity_floor": 1 / (max(len(r) for r in S.ref.values()) + 1)}
    return Result(out, seg, th, counts)
