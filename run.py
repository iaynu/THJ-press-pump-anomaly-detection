"""KAMP press hydraulic pump - anomaly early detection. One entry point for everything.

  python run.py train                       fit the final system on all normal data -> models/
  python run.py predict --input FILE.csv    write row-level predictions -> results/predictions_<name>.csv
  python run.py evaluate                    all experiments -> results/raw, results/tables, results/figures (~50 min, 2 cores)
  python run.py all                         train + predict on the provided files + evaluate

Input CSV format = the competition files: index, TimeStamp, AI0_Vibration, AI1_Vibration, AI2_Current
(Equipment_state optional). No timestamp, date or gap information is used as a model input.
"""
import argparse
import json
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
warnings.filterwarnings("ignore")
from kamp.data import load_segments                                     # noqa: E402
from kamp.models import final_system, VIBONLY, CURRENT                   # noqa: E402
from kamp.analysis import analyze, read_csv                               # noqa: E402
from kamp.evaluate import _ewma                                          # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA, MODELS, RESULTS = (os.path.join(ROOT, d) for d in ("data", "models", "results"))


def train():
    os.makedirs(MODELS, exist_ok=True)
    N = load_segments(os.path.join(DATA, "press_data_normal.csv"), 0)
    xs = [s.x for s in N]
    S = final_system().fit(xs, seed=0)
    # trend (EWMA) rule calibrated on the out-of-fold stream in time order: strict single q99.7 OR EWMA, 1% target
    order = np.argsort([s.t0 for s in N])
    oof = S.oof[order]
    z_ref = S.z(oof[~np.isnan(oof)])
    strict = float(np.quantile(z_ref, 0.997))
    er = _ewma(z_ref)
    cands = np.quantile(er, np.linspace(0.99, 0.9999, 60))
    e_th = float(next((c for c in cands if np.mean((z_ref > strict) | (er > c)) <= 0.01), cands[-1]))
    bundle = {"system": S, "vib_diag": VIBONLY().fit(xs), "cur_diag": CURRENT().fit(xs),
              "trend": {"strict_z": strict, "ewma_z": e_th, "lambda": 0.2}}
    with open(os.path.join(MODELS, "final_system.pkl"), "wb") as f:
        pickle.dump(bundle, f)
    info = {"normal_segments": len(N), "robust_trim_z": S.trim_z, "trimmed_training_share": float(S.trimmed_share), "attention_threshold_q95": float(S.t_att), "warning_threshold_q99": float(S.t_warn),
            "trend_rule": bundle["trend"], "views": list(S.views)}
    with open(os.path.join(MODELS, "final_system.json"), "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print("trained:", json.dumps(info, ensure_ascii=False))


def predict(path, out=None):
    """row-level predictions for EVERY input row (see kamp.analysis for row_status and short segments)"""
    model_path = os.path.join(MODELS, "final_system.pkl")
    if not os.path.exists(model_path):
        raise SystemExit(f"{model_path} 이(가) 없습니다. 먼저 `python run.py train`을 실행하세요.")
    with open(model_path, "rb") as f:
        B = pickle.load(f)
    R = analyze(read_csv(path), B)
    P = R.rows.rename(columns={"rarity": "rarity_1_minus_p"})
    keep = ["TimeStamp"] + (["Equipment_state"] if "Equipment_state" in P else []) + [
        "row_status", "segment_id", "segment_rows", "judged_directly", "p_joint", "p_current", "normal_percentile",
        "rarity_1_minus_p", "level", "level_name", "trend_warning", "pred_label", "diagnosis"]
    P = P[keep]
    P.index.name = "row"
    os.makedirs(RESULTS, exist_ok=True)
    out = out or os.path.join(RESULTS, f"predictions_{os.path.splitext(os.path.basename(path))[0]}.csv")
    P.to_csv(out, encoding="utf-8-sig")
    c = R.counts
    print(f"{os.path.basename(path)}: rows {c['rows_total']} (analysed {c['rows_analyzed']}, duplicate {c['rows_duplicate']}, "
          f"excluded {c['rows_invalid']}), segments {c['segments']} (normal {c['seg_normal']}, attention {c['seg_attention']}, "
          f"warning {c['seg_warning']}), rows predicted abnormal {int(P.pred_label.sum())} -> {out}")


def evaluate():
    """all experiments; two at a time (the machine this was built on had 2 cores)"""
    import subprocess
    if not os.path.exists(os.path.join(DATA, "press_data_normal.csv")):
        raise SystemExit("data/press_data_normal.csv 가 없습니다.")
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))

    def pair(a, b):
        p = subprocess.Popen([sys.executable, "-m", *a], cwd=ROOT, env=env)
        subprocess.run([sys.executable, "-m", *b], cwd=ROOT, env=env, check=True)
        if p.wait():
            raise SystemExit(f"failed: {a}")

    pair(["kamp.evaluate", "random"], ["kamp.evaluate", "blocks"])          # model comparison (~25 min)
    pair(["kamp.evaluate", "forward"], ["kamp.sensitivity"])                # forward split, threshold sensitivity
    pair(["kamp.trim_sweep", "clean"], ["kamp.trim_sweep", "contam"])       # robust calibration: cost and benefit
    pair(["kamp.trim_sweep", "contam_m3"], ["kamp.contamination"])          # label noise: M3, and M2 vs M3 without trimming
    subprocess.run([sys.executable, "-m", "kamp.regime_check"], cwd=ROOT, env=env, check=True)   # low-load state check
    subprocess.run([sys.executable, "-m", "kamp.report"], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "predict", "evaluate", "all"])
    ap.add_argument("--input")
    ap.add_argument("--output")
    a = ap.parse_args()
    if a.cmd == "predict" and not a.input:
        ap.error("predict에는 --input <CSV 경로>가 필요합니다.")
    if a.cmd in ("train", "all"):
        train()
    if a.cmd == "predict":
        predict(a.input, a.output)
    if a.cmd == "all":
        for f in ("outlier_data.csv", "press_data_normal.csv"):
            predict(os.path.join(DATA, f))
        evaluate()
    if a.cmd == "evaluate":
        evaluate()
