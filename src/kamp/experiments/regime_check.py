"""Follow-up on the main false-negative condition: vibration faults inside the low-load state are missed.
Does a mixture model (Gaussian mixture on the same 9 features, one component per operating state)
recover them without raising false alarms? Random segment 5-fold, seed 0. Writes results/raw/regime_check.csv"""
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")
from ..data import load_segments
from ..detectors import clear_cache
from ..faults import FAULTS
from ..models import catalogue, M2, M3
from ..system import fill_short
from .evaluate import PICK, low_load


def main():
    CAT = catalogue()
    MODELS = {M2: CAT[M2], M3: CAT[M3]}
    N = load_segments("data/press_data_normal.csv", 0)
    O = load_segments("data/outlier_data.csv", 1)
    rows = []
    for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(np.arange(len(N)))):
        rng = np.random.default_rng(fold)
        xs = [N[i].x for i in tr]
        test = [N[i] for i in te]
        long = [s for s in test if s.n >= 16]
        sd = np.concatenate([x[:, :2] for x in xs]).std(0)
        inj = {f"{f} {v}": [FAULTS[f][2](s.x, v, sd, rng) for s in long] for f, v in PICK}
        ll = np.array([low_load(s) for s in long])
        for name, mk in MODELS.items():
            S = mk().fit(xs)
            lv, _ = S.levels([s.x for s in test])
            lv = fill_short(lv, [s.t0 for s in test])
            lo, _ = S.levels([s.x for s in O])
            r = dict(fold=fold, model=name, fa=np.mean(lv == 2), fa_lowload=np.mean(lv[[low_load(s) for s in test]] == 2),
                     real=np.mean(fill_short(lo, [s.t0 for s in O]) == 2))
            for k, xx in inj.items():
                l2, _ = S.levels(xx)
                r[k] = np.mean(l2 == 2)
                r[k + " | 저부하"] = np.mean(l2[ll] == 2)
            rows.append(r)
        clear_cache()
        print("fold", fold, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv("results/raw/regime_check.csv", index=False)
    cols = [f"{f} {v}" for f, v in PICK]
    d["고장 평균"] = d[cols].mean(1)
    d["저부하 고장 평균"] = d[[c + " | 저부하" for c in cols]].mean(1)
    pd.set_option("display.width", 250)
    print(d.groupby("model")[["fa", "fa_lowload", "real", "고장 평균", "저부하 고장 평균",
                              "진동 전체 증가 1.5 | 저부하", "진동 전체 증가 2.0 | 저부하", "하부 진동 극성 반전 1 | 저부하"]].mean().round(3).T.to_string())


if __name__ == "__main__":
    main()
