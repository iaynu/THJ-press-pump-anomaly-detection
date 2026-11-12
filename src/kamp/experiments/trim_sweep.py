"""How strict should the robust (trimmed) threshold calibration be?
For trim_z in {none, 3, 3.5, 4, 5, 6}: clean-data false alarms (random / blocks / forward), fault recall,
and recall of the same early fault when part of the normal training label is contaminated with it.
The detectors are fitted once per training set; only the calibration step changes with trim_z.
  python -m kamp.experiments.trim_sweep clean    -> results/raw/trim_sweep_clean.csv
  python -m kamp.experiments.trim_sweep contam   -> results/raw/trim_sweep_contam.csv
  python -m kamp.experiments.trim_sweep contam_m3 -> results/raw/trim_sweep_contam_m3.csv (same check for the mixture model M3)"""
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")
from ..data import load_segments
from ..detectors import clear_cache
from ..faults import FAULTS, spikes, scale
from ..models import JOINT, JOINT_GMM, CURRENT
from ..system import System, fill_short, oof_scores, _pct
from .evaluate import PICK

TRIMS = [None, 3.0, 3.5, 4.0, 5.0, 6.0]
N = load_segments("data/press_data_normal.csv", 0)
O = load_segments("data/outlier_data.csv", 1)


def fit_raw(xs, joint=JOINT):
    S = System({"joint": joint, "current": CURRENT}, fusion="max")
    raw = {n: oof_scores(f, xs, seed=0) for n, f in S.views.items()}
    S.det = {n: f().fit(xs) for n, f in S.views.items()}
    return S, raw


def levels_for(S, tz, raw, test_raw):
    S.trim_z = tz
    S.calibrate(raw)
    out = []
    for r in test_raw:
        VP = np.column_stack([_pct(S.ref[n], r[n]) for n in S.views])
        out.append(S.levels(None, VP)[0])
    return out


def raw_scores(S, xs):
    return {n: S.det[n].seg_scores(xs) for n in S.views}


def clean():
    rows = []
    order = np.argsort([s.t0 for s in N])
    splits = [("random", f, tr, te) for f, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(np.arange(len(N))))]
    splits += [("blocks", f, np.array([i for i in order if i not in set(b)]), b) for f, b in enumerate(np.array_split(order, 5))]
    cut = int(0.8 * len(order))
    splits += [("forward", 0, order[:cut], order[cut:])]
    for proto, fold, tr, te in splits:
        rng = np.random.default_rng(fold)
        xs = [N[i].x for i in tr]
        test = [N[i] for i in te]
        S, raw = fit_raw(xs)
        sd = np.concatenate([x[:, :2] for x in xs]).std(0)
        long = [s.x for s in test if s.n >= 16]
        faults = [[FAULTS[f][2](x, v, sd, rng) for x in long] for f, v in PICK]
        test_raw = [raw_scores(S, [s.x for s in test]), raw_scores(S, [s.x for s in O])] + [raw_scores(S, F) for F in faults]
        for tz in TRIMS:
            L = levels_for(S, tz, raw, test_raw)
            rows.append(dict(protocol=proto, fold=fold, trim=str(tz), trimmed=S.trimmed_share,
                             false_alarm=np.mean(fill_short(L[0], [s.t0 for s in test]) == 2),
                             real_recall=np.mean(fill_short(L[1], [s.t0 for s in O]) == 2),
                             fault_recall=np.mean([np.nanmean(l == 2) for l in L[2:]])))
        clear_cache()
        print(proto, fold, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv("results/raw/trim_sweep_clean.csv", index=False)
    print(d.groupby(["protocol", "trim"], sort=False)[["trimmed", "false_alarm", "fault_recall", "real_recall"]].mean().round(4).to_string())


def contam(joint=JOINT, out="results/raw/trim_sweep_contam.csv"):
    rows = []
    for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(np.arange(len(N)))):
        if fold >= 3:
            break
        rng = np.random.default_rng(fold)
        sd = np.concatenate([N[i].x[:, :2] for i in tr]).std(0)
        test = [N[i] for i in te]
        ef = lambda x: spikes(scale(x, 1.5, [0, 1]), 3, sd, rng)
        faulty = [ef(s.x) for s in test if s.n >= 16]
        for p in [0.02, 0.05, 0.1, 0.2]:
            bad = set(rng.permutation(tr)[: int(round(p * len(tr)))])
            xs = [ef(N[i].x) if i in bad else N[i].x for i in tr]
            S, raw = fit_raw(xs, joint)
            test_raw = [raw_scores(S, [s.x for s in test]), raw_scores(S, faulty), raw_scores(S, [s.x for s in O])]
            for tz in TRIMS:
                L = levels_for(S, tz, raw, test_raw)
                rows.append(dict(fold=fold, share=p, trim=str(tz), trimmed=S.trimmed_share,
                                 false_alarm=np.mean(fill_short(L[0], [s.t0 for s in test]) == 2),
                                 same_fault_recall=np.nanmean(L[1] == 2),
                                 real_recall=np.mean(fill_short(L[2], [s.t0 for s in O]) == 2)))
            clear_cache()
        print("fold", fold, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(out, index=False)
    print(d.groupby(["share", "trim"], sort=False)[["trimmed", "false_alarm", "same_fault_recall", "real_recall"]].mean().round(3).to_string())


if __name__ == "__main__":
    {"clean": clean, "contam": contam,
     "contam_m3": lambda: contam(JOINT_GMM, "results/raw/trim_sweep_contam_m3.csv")}[sys.argv[1]]()
