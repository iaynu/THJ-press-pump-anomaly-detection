"""Our models under the Colab notebook's own protocol, scored on the same tests the notebook reports.

Notebook protocol (reproduced here): normal segments in time order -> first 60% train, next 20% calibration,
last 20% test. Our models are fitted on the same TRAIN segments only (their own out-of-fold calibration replaces the
notebook's calibration subset). Test = the same last-20% normal segments.
Tests: false warnings on normal test segments, recall on the provided abnormal recording, the notebook's five
synthetic perturbations, its gain / offset / order-shuffle / circular-shift probes, and our 11 injected faults.
Detection power is also compared threshold-free: AUC of each fault against the same segments left clean.
Writes results/notebook_compare/*.csv
  python -m kamp.notebook_compare
The notebook's own numbers come from tables written by running it (see results/notebook_compare/README.txt)."""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
from .data import load_segments
from .detectors import clear_cache
from .faults import FAULTS
from .models import catalogue, M3, FINAL
from .system import fill_short
from .evaluate import PICK

OUT = "results/notebook_compare"
MODELS = {"M1 통합 모델 (9특징 MCD)": "M1 통합 모델 (9특징 MCD)", "최종 (M2 + 견고한 보정)": FINAL, "M3 상태 혼합": M3}
THEIR_PERTURB = ["vibration_scale", "vibration_impulse", "current_center_shift", "current_amplitude_increase", "current_distortion"]


def perturb(x, kind, std, route):
    """the notebook's synthetic perturbations (sizes: x1.5, +4 std impulse, +0.5 std centre, x1.3, 0.3 std 2 Hz), at segment level.
    The notebook adds the impulse at the middle of every 50%-overlap window; here at every half-window step."""
    x = x.copy()
    t = np.arange(len(x)) * 0.1
    if kind == "vibration_scale":
        x[:, :2] *= 1.5
    elif kind == "vibration_impulse":
        step = max(1, route // 2)
        for p in range(step, len(x) - step + 1, step):
            x[p, :2] += 4.0 * std[:2]
    elif kind == "current_center_shift":
        x[:, 2] += 0.5 * std[2]
    elif kind == "current_amplitude_increase":
        x[:, 2] = x[:, 2].mean() + 1.3 * (x[:, 2] - x[:, 2].mean())
    elif kind == "current_distortion":
        x[:, 2] += 0.3 * std[2] * np.sin(2 * np.pi * 2 * t)
    return x


def margin(S, xs):
    """threshold-free score: max over views of raw score / that view's own q99 cutoff (>1 means a warning-level view)"""
    m = np.column_stack([S.det[n].seg_scores(xs) / np.quantile(S.ref[n], 0.99) for n in S.views])
    return np.where(np.isnan(m).any(1), np.nan, m.max(1))


def main():
    os.makedirs(OUT, exist_ok=True)
    N = load_segments("data/press_data_normal.csv", 0)
    O = load_segments("data/outlier_data.csv", 1)
    order = np.argsort([s.t0 for s in N])
    tr, te = order[:359], order[479:]            # notebook: 360 / 120 / 120 (it finds one more early segment)
    xs_tr = [N[i].x for i in tr]
    test = [N[i] for i in te]
    rows_tr = np.vstack(xs_tr)
    std = rows_tr.std(0)
    sd2 = rows_tr[:, :2].std(0)
    cat = catalogue()
    out = {"table": [], "auc": [], "falsify": [], "perturb": [], "faults": []}
    X_te = [s.x for s in test]
    direct = np.array([len(x) >= 8 for x in X_te])
    long = np.array([len(x) >= 16 for x in X_te])
    rng_state = lambda: np.random.default_rng(0)
    for label, key in MODELS.items():
        S = cat[key]().fit(xs_tr)
        route = lambda x: S.det["joint"].route(x)

        def lv(xs):
            return S.levels(xs)[0]

        clean_lv, clean_m = lv(X_te), margin(S, X_te)
        real_lv, real_m = lv([s.x for s in O]), margin(S, [s.x for s in O])
        ok_real = ~np.isnan(real_lv)
        fa, att = np.nanmean(clean_lv == 2), np.nanmean(clean_lv >= 1)
        tp, fn = int((real_lv[ok_real] == 2).sum()), int((real_lv[ok_real] < 2).sum())
        fp = int(np.nansum(clean_lv == 2))
        tn = int(np.nansum(clean_lv < 2))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)
        out["table"].append(dict(model=label, false_warning_segment_rate=fa, attention_segment_rate=att,
                                 n_direct_normal=int(direct.sum()), abnormal_direct_recall=rec, abnormal_all_recall=np.mean(
                                     fill_short(real_lv, [s.t0 for s in O]) == 2),
                                 precision_mixture=prec, f1_mixture=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                                 tp=tp, fn=fn, fp=fp, tn=tn,
                                 real_vs_normal_auc=roc_auc_score(np.r_[np.zeros(direct.sum()), np.ones(ok_real.sum())],
                                                                  np.r_[clean_m[direct], real_m[ok_real]])))

        def evaluate(name, fn_, mask, group, store):
            xs = [fn_(x) for x, k in zip(X_te, mask) if k]
            l_, m_ = lv(xs), margin(S, xs)
            c_m = clean_m[mask]
            ok = ~np.isnan(m_)
            store.append(dict(model=label, test=name, n=int(ok.sum()), warning_rate=np.nanmean(l_ == 2),
                              attention_rate=np.nanmean(l_ >= 1),
                              auc_vs_clean=roc_auc_score(np.r_[np.zeros(ok.sum()), np.ones(ok.sum())], np.r_[c_m[ok], m_[ok]])))

        rng = rng_state()
        for kind in THEIR_PERTURB:
            evaluate(kind, lambda x, kind=kind: perturb(x, kind, std, route(x) or 8), direct, "perturb", out["perturb"])
        for name, fn_ in [("gain_0.8", lambda x: x * 0.8), ("gain_1.2", lambda x: x * 1.2), ("offset", lambda x: x + 0.2 * std),
                          ("sample_order_shuffle", lambda x: x[rng.permutation(len(x))]),
                          ("circular_shift", lambda x: np.roll(x, max(1, len(x) // 3), axis=0))]:
            evaluate(name, fn_, direct, "falsify", out["falsify"])
        for f, v in PICK:
            r = np.random.default_rng(0)
            evaluate(f"{f} {v}", lambda x, f=f, v=v: FAULTS[f][2](x, v, sd2, r), long, "faults", out["faults"])
        clear_cache()
        print("done", label, flush=True)
    pd.DataFrame(out["table"]).to_csv(f"{OUT}/ours_notebook_protocol.csv", index=False)
    for k in ("perturb", "falsify", "faults"):
        pd.DataFrame(out[k]).to_csv(f"{OUT}/ours_{k}.csv", index=False)
    print(pd.DataFrame(out["table"]).round(3).T.to_string())


if __name__ == "__main__":
    main()
