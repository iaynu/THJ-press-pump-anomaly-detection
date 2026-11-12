"""Two interventions on the notebook's final model (joint 9-feature MCD, max over windows), run inside its namespace:
  model:     pooled (one MCD for 24/16/8-row windows, as in the notebook)  vs  split (one MCD per window length)
  threshold: nb (notebook: window q99 on its calibration subset)  |  oof_global (segment q99 of out-of-fold TRAIN scores, one cutoff)
             |  oof_route (same, one cutoff per window length)
Everything else (features, split, test segments, abnormal file) is untouched."""
import os, sys, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
exec(compile(open(os.path.join(HERE, "run_nb.py")).read(), "run_nb.py", "exec"), globals())
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(ROOT, "src"))
from kamp.faults import FAULTS
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

PICK = [("진동 전체 증가", 1.5), ("진동 전체 증가", 2.0), ("충격 스파이크(16행당 1회)", 2), ("충격 스파이크(16행당 1회)", 3),
        ("하부 진동 극성 반전", 1), ("전류 오프셋 이동", 5), ("전류 오프셋 이동", 10), ("전류 진폭 증가", 1.2),
        ("전류 진폭 증가", 1.3), ("전류 파형 왜곡(2.5Hz 성분)", 0.05), ("전류 파형 왜곡(2.5Hz 성분)", 0.1)]
S0, S1 = SENSOR_COLUMNS[0], SENSOR_COLUMNS[1]
PF = PHYSICS_FEATURES
print("train windows by n:", physics_train.window_n.value_counts().to_dict())

# ---- models
split_ref = {k: fit_mcd_reference(physics_train[physics_train.window_n.eq(k)], PF) for k in sorted(physics_train.window_n.unique())}
def win_scores(model, df, pool=None, sp=None):
    pool, sp = pool or joint_reference, sp or split_ref
    if model == "pooled": return mcd_score(pool, df)
    out = np.full(len(df), np.nan); n = df.window_n.to_numpy()
    for k, ref in sp.items():
        m = n == k
        if m.any(): out[m] = mcd_score(ref, df.loc[m])
    return out

# ---- thresholds
cal = physics.loc[physics.split.eq("calibration") & physics.calibration_role.eq("cutoff_fit")]
thr = {}
for model in ("pooled", "split"):
    s = pd.Series(win_scores(model, cal), index=cal.index)
    if model == "pooled": thr[(model, "nb")] = {k: float(np.quantile(s, .99)) for k in (8, 16, 24)}
    else: thr[(model, "nb")] = {k: float(np.quantile(s[cal.window_n.eq(k)], .99)) for k in (8, 16, 24)}
segs = physics_train.segment_id.unique()
oof = {"pooled": [], "split": []}
for tri, tei in KFold(5, shuffle=True, random_state=0).split(segs):
    a = physics_train[physics_train.segment_id.isin(segs[tri])]; b = physics_train[physics_train.segment_id.isin(segs[tei])]
    pool = fit_mcd_reference(a, PF); sp = {k: fit_mcd_reference(a[a.window_n.eq(k)], PF) for k in sorted(a.window_n.unique())}
    for model in oof:
        sc = win_scores(model, b, pool, sp)
        oof[model].append(b.assign(s=sc).groupby("segment_id").agg(smax=("s", "max"), n=("window_n", "first")))
for model in oof:
    o = pd.concat(oof[model])
    thr[(model, "oof_global")] = {k: float(np.quantile(o.smax, .99)) for k in (8, 16, 24)}
    thr[(model, "oof_route")] = {k: float(np.quantile(o.smax[o.n.eq(k)], .99)) for k in (8, 16, 24)}
    print(model, "OOF segment scores by n (median / q99 / max):", {k: tuple(np.round([o.smax[o.n.eq(k)].median(), o.smax[o.n.eq(k)].quantile(.99), o.smax[o.n.eq(k)].max()], 1)) for k in (8, 16, 24)}, "n segs", o.n.value_counts().to_dict())
pd.DataFrame([{"model": m, "rule": r, **{f"cutoff_n{k}": v for k, v in d.items()}} for (m, r), d in thr.items()]).to_csv(os.path.join(HERE, "..", "variants", "variants_cutoffs.csv"), index=False)
print(pd.read_csv(os.path.join(HERE, "..", "variants", "variants_cutoffs.csv")).round(1).to_string())
VARIANTS = list(thr)

def seg_table(df):
    """per segment, per variant: warning flag and margin (max window score / cutoff)"""
    rows = []
    sc = {m: win_scores(m, df) for m in ("pooled", "split")}
    for (m, r) in VARIANTS:
        g = df[["segment_id", "window_n"]].assign(s=sc[m]).groupby("segment_id").agg(smax=("s", "max"), n=("window_n", "first"))
        cut = g.n.map(thr[(m, r)])
        rows.append(pd.DataFrame({"segment_id": g.index, "model": m, "rule": r, "margin": (g.smax / cut).to_numpy(), "warning": (g.smax > cut).to_numpy()}))
    return pd.concat(rows)

# ---- test sets
normal_test_w = physics.loc[physics.split.eq("normal_test")]
abn_w = physics.loc[physics.split.eq("abnormal_evaluation")]
out = {"recorded": [], "tests": []}
tn, ta = seg_table(normal_test_w), seg_table(abn_w)
for (m, r) in VARIANTS:
    a, b = tn[(tn.model == m) & (tn.rule == r)], ta[(ta.model == m) & (ta.rule == r)]
    tp, fp = int(b.warning.sum()), int(a.warning.sum()); prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / len(b)
    out["recorded"].append(dict(model=m, rule=r, normal_test_segments=len(a), false_warnings=fp, false_warning_rate=fp / len(a),
        abnormal_direct_segments=len(b), abnormal_recall=rec, precision=prec, f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        real_vs_normal_auc=roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a.margin, b.margin])))

def feats_of(base, transform):
    recs = []
    for _, row in base.iterrows():
        v = transform(waveforms[row.window_id].copy(), row, window_times[row.window_id])
        r = row.to_dict(); r.update(physics_vector(v, window_times[row.window_id])); recs.append(r)
    return pd.DataFrame(recs)

def record(group, name, changed, clean):
    sc, sl = seg_table(changed), seg_table(clean)
    for (m, r) in VARIANTS:
        a = sc[(sc.model == m) & (sc.rule == r)].set_index("segment_id"); c = sl[(sl.model == m) & (sl.rule == r)].set_index("segment_id").loc[a.index]
        out["tests"].append(dict(group=group, test=name, model=m, rule=r, n=len(a), warning_rate=a.warning.mean(),
            auc_vs_clean=roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(a))], np.r_[c.margin, a.margin])))

base = falsification_base
rng = np.random.default_rng(RANDOM_STATE)
for kind in SYNTHETIC_KINDS:
    record("notebook_perturbation", kind, feats_of(base, lambda x, row, t, kind=kind: perturb(x, row, t, kind)), base)
probes = {"gain_0.8": lambda x, row, t: x * 0.8, "gain_1.2": lambda x, row, t: x * 1.2, "offset": lambda x, row, t: x + offset_vector,
          "sample_order_shuffle": lambda x, row, t: x[rng.permutation(len(x))], "circular_shift": lambda x, row, t: np.roll(x, max(1, len(x) // 3), axis=0)}
for name, fn in probes.items(): record("falsification", name, feats_of(base, fn), base)

test_rows = normal_analysis[normal_analysis.split.eq("normal_test")]
sd = train_rows[[S0, S1]].std().to_numpy()
def faulted(fn, rng):
    frames = []
    for sid, g in test_rows.groupby("segment_id", sort=False):
        if len(g) < 16: continue
        y = fn(g[[S0, S1, "AI2_analysis"]].to_numpy(float), rng)
        g2 = g.copy(); g2[S0] = y[:, 0]; g2[S1] = y[:, 1]; g2["AI2_analysis"] = y[:, 2]; frames.append(g2)
    return make_windows(pd.concat(frames))[0]
clean16 = faulted(lambda x, r: x, np.random.default_rng(0))
for f, v in PICK:
    record("injected_fault", f"{f} {v}", faulted(lambda x, r, f=f, v=v: FAULTS[f][2](x, v, sd, r), np.random.default_rng(0)), clean16)

pd.DataFrame(out["recorded"]).to_csv(os.path.join(HERE, "..", "variants", "variants_recorded.csv"), index=False)
pd.DataFrame(out["tests"]).to_csv(os.path.join(HERE, "..", "variants", "variants_tests.csv"), index=False)
print("VARIANTS DONE")
