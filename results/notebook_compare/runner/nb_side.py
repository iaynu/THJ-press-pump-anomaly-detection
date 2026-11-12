"""Runs inside the notebook's own namespace (after executing it) and scores OUR injected faults with ITS final model
(A_physics / none: joint 9-feature MCD, window q99 cutoff from its calibration subset, segment warns if any window warns)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
exec(compile(open(os.path.join(HERE, "run_nb.py")).read(), "run_nb.py", "exec"), globals())
sys.path.insert(0, os.path.join(ROOT, "src"))
from kamp.faults import FAULTS, spikes, scale

PICK = [("진동 전체 증가", 1.5), ("진동 전체 증가", 2.0), ("충격 스파이크(16행당 1회)", 2), ("충격 스파이크(16행당 1회)", 3),
        ("하부 진동 극성 반전", 1), ("전류 오프셋 이동", 5), ("전류 오프셋 이동", 10), ("전류 진폭 증가", 1.2),
        ("전류 진폭 증가", 1.3), ("전류 파형 왜곡(2.5Hz 성분)", 0.05), ("전류 파형 왜곡(2.5Hz 성분)", 0.1)]
S0, S1 = SENSOR_COLUMNS[0], SENSOR_COLUMNS[1]
test_rows = normal_analysis[normal_analysis.split.eq("normal_test")]
sd = train_rows[[S0, S1]].std().to_numpy()
cut = joint = thresholds["joint_score"][str(WARNING_QUANTILE)]

def run(fn, rng):
    frames = []
    for sid, g in test_rows.groupby("segment_id", sort=False):
        if len(g) < 16: continue
        y = fn(g[[S0, S1, "AI2_analysis"]].to_numpy(float), rng)
        g2 = g.copy(); g2[S0] = y[:, 0]; g2[S1] = y[:, 1]; g2["AI2_analysis"] = y[:, 2]
        frames.append(g2)
    feats, waves, times = make_windows(pd.concat(frames))
    sc = score_components(feats, waves, times)
    win = alarm_windows(sc, FINAL_CANDIDATE, FINAL_POLICY)
    seg = win.groupby("segment_id").agg(warning=("warning", "max"), attention=("attention", "max"), margin=("candidate_score", "max"))
    return seg.reset_index()

rows = []
base = run(lambda x, r: x, np.random.default_rng(0)); base["kind"] = "정상"; rows.append(base)
for f, v in PICK:
    rng = np.random.default_rng(0)
    d = run(lambda x, r, f=f, v=v: FAULTS[f][2](x, v, sd, r), rng); d["kind"] = f"{f} {v}"; rows.append(d)
out = pd.concat(rows)
out.to_csv(os.path.join(HERE, "..", "notebook_outputs", "theirs_on_our_faults.csv"), index=False)
print(out.groupby("kind").agg(n=("warning", "size"), warn=("warning", "mean"), att=("attention", "mean"), med_margin=("margin", "median")).round(3).to_string())
# their real-anomaly and normal-test segment margins for AUC
seg = final_segments[["segment_id", "source", "split", "direct_evidence", "candidate_score", "warning", "attention", "segment_n_samples"]]
seg.to_csv(os.path.join(HERE, "..", "notebook_outputs", "theirs_segments.csv"), index=False)
