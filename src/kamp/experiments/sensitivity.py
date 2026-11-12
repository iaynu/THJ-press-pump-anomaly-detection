"""Threshold sensitivity of the final system: how false alarms, fault recall and real-anomaly recall
move as the warning quantile changes (q = 0.90 ... 0.995 of out-of-fold normal scores).
Random segment 5-fold, seed 0. Writes results/raw/sensitivity.csv"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
from ..data import load_segments
from ..faults import FAULTS
from ..models import final_system
from ..system import fill_short
from .evaluate import PICK
from ..detectors import clear_cache

QS = [0.90, 0.925, 0.95, 0.97, 0.98, 0.99, 0.995]


def main(out_dir="results/raw", data_dir="data"):
    N = load_segments(os.path.join(data_dir, "press_data_normal.csv"), 0)
    O = load_segments(os.path.join(data_dir, "outlier_data.csv"), 1)
    xo, to = [s.x for s in O], [s.t0 for s in O]
    rows = []
    for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(np.arange(len(N)))):
        rng = np.random.default_rng(fold)
        xs = [N[i].x for i in tr]
        test = [N[i] for i in te]
        S = final_system().fit(xs, seed=0)
        oof = S.oof[~np.isnan(S.oof)]
        s_n = fill_short(S.score([s.x for s in test]), [s.t0 for s in test])
        s_o = fill_short(S.score(xo), to)
        clean = [s.x for s in test if s.n >= 16]
        sd = np.concatenate([x[:, :2] for x in xs]).std(0)
        s_f = {f"{f} {v}": S.score([FAULTS[f][2](x, v, sd, rng) for x in clean]) for f, v in PICK}
        for q in QS:
            t = np.quantile(oof, q)
            rows.append(dict(fold=fold, q=q, false_alarm=np.mean(s_n >= t), real_recall=np.mean(s_o >= t),
                             fault_recall=np.mean([np.nanmean(v >= t) for v in s_f.values()]),
                             alarms_per_100=100 * np.mean(s_n >= t)))
        clear_cache()
        print("fold", fold, flush=True)
    pd.DataFrame(rows).to_csv(f"{out_dir}/sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
