"""Label-noise check for the two final candidates (M2 MCD joint vs M3 mixture joint).
Part of the 'normal' training label is replaced by an early-stage fault (vibration x1.5 + 3-sigma impacts)
or by real abnormal segments; we measure whether the model learns them as normal and stops detecting them.
Random segment 5-fold, seed 0. Writes results/raw/contamination.csv"""
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")
from .data import load_segments
from .detectors import clear_cache
from .faults import spikes, scale
from .models import catalogue, M2, M3
from .system import fill_short

N = load_segments("data/press_data_normal.csv", 0)
O = load_segments("data/outlier_data.csv", 1)
CAT = catalogue()


def early_fault(x, sd, rng):
    return spikes(scale(x, 1.5, [0, 1]), 3, sd, rng)


rows = []
for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(np.arange(len(N)))):
    rng = np.random.default_rng(fold)
    sd = np.concatenate([N[i].x[:, :2] for i in tr]).std(0)
    test = [N[i] for i in te]
    long = [s.x for s in test if s.n >= 16]
    faulty = [early_fault(x, sd, rng) for x in long]
    for p in [0.0, 0.1, 0.2]:
        bad = set(rng.permutation(tr)[: int(round(p * len(tr)))])
        xs = [early_fault(N[i].x, sd, rng) if i in bad else N[i].x for i in tr]
        for name in (M2, M3):
            S = CAT[name]().fit(xs)
            lv, _ = S.levels([s.x for s in test])
            lf, _ = S.levels(faulty)
            lo, _ = S.levels([s.x for s in O])
            rows.append(dict(fold=fold, share=p, model=name, false_alarm=np.mean(fill_short(lv, [s.t0 for s in test]) == 2),
                             same_fault_recall=np.nanmean(lf == 2), real_recall=np.mean(fill_short(lo, [s.t0 for s in O]) == 2)))
        clear_cache()
    print("fold", fold, flush=True)
d = pd.DataFrame(rows)
d.to_csv("results/raw/contamination.csv", index=False)
print(d.groupby(["model", "share"])[["false_alarm", "same_fault_recall", "real_recall"]].mean().round(3).to_string())
