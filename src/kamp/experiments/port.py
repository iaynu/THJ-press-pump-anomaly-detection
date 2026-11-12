"""Which ideas from the Colab notebook help OUR model? One change at a time on our final structure
(joint 9-feature MCD + current-only MCD, max fusion, robust calibration), random segment 5-fold (seed 0).
Ideas taken from the notebook:
  P1/P2 segment score = max / top-2 mean of the window scores (ours: mean)
  P3    MCD support fraction 0.75 (ours 0.9)
  P4    robust scaler: median / IQR (ours: mean / std)
  P5    current distortion as absolute residual RMS (ours: residual / amplitude)
  P6    windows with 50% overlap (ours: step 4 rows)
Metrics: false warnings, recall on the provided abnormal recording, injected-fault recall (11 faults, also split into
low-load / other operating state), 1-sample glitch warnings, gain x1.2 warnings.
Writes results/raw/port.csv"""
import sys
import warnings
import numpy as np
import pandas as pd
from functools import partial
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")
from ..data import load_segments
from ..detectors import FeatureDetector, MCD, clear_cache
from ..faults import FAULTS, glitch
from ..features import CUR
from ..system import System, fill_short
from .evaluate import PICK, low_load

VARIANTS = {
    "P0 기준 (우리 최종 구조)": {},
    "P1 구간 점수 = 윈도우 최댓값": dict(agg="max"),
    "P2 구간 점수 = 윈도우 상위 2개 평균": dict(agg="top2"),
    "P3 MCD 지지 비율 0.75": dict(support=0.75),
    "P4 로버스트 스케일러 (중앙값/IQR)": dict(scaler="robust"),
    "P5 전류 왜곡 = 절대 잔차": dict(abs_distortion=True),
    "P6 윈도우 50% 겹침": dict(half_overlap=True),
}


def build(support=0.9, scaler="std", **fd):
    mcd = partial(MCD, support, scaler)
    return lambda: System({"joint": partial(FeatureDetector, mcd, None, name="joint", **fd),
                           "current": partial(FeatureDetector, mcd, CUR, name="current", **fd)}, fusion="max", trim_z=3.5)


def main(names=None, seeds=(0,), out="results/raw/port.csv", variants=None):
    N = load_segments("data/press_data_normal.csv", 0)
    O = load_segments("data/outlier_data.csv", 1)
    rows = []
    variants = variants or {k: v for k, v in VARIANTS.items() if names is None or k in names}
    for seed in seeds:
        for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=seed).split(np.arange(len(N)))):
            rng = np.random.default_rng(1000 * seed + fold)
            xs = [N[i].x for i in tr]
            test = [N[i] for i in te]
            tx, tt = [s.x for s in test], [s.t0 for s in test]
            xo, to = [s.x for s in O], [s.t0 for s in O]
            li = [i for i, s in enumerate(test) if s.n >= 16]
            clean = [tx[i] for i in li]
            ll = np.array([low_load(test[i]) for i in li])
            sd = np.concatenate([x[:, :2] for x in xs]).std(0)
            faulty = {f"{f} {v}": [FAULTS[f][2](x, v, sd, rng) for x in clean] for f, v in PICK}
            gl = [glitch(x, rng) for x in clean]
            g12 = [x * 1.2 for x in clean]
            gs = [glitch(x, rng, 0.25) for x in clean]
            g2 = [glitch(glitch(x, rng), rng) for x in clean]
            for name, kw in variants.items():
                S = build(**kw)().fit(xs, seed=seed)
                lv_n, _ = S.levels(tx)
                lv_o, _ = S.levels(xo)
                r = dict(seed=seed, fold=fold, model=name,
                         false_alarm=np.mean(fill_short(lv_n, tt) == 2), real_recall=np.mean(fill_short(lv_o, to) == 2),
                         glitch=np.nanmean(S.levels(gl)[0] == 2), glitch_small=np.nanmean(S.levels(gs)[0] == 2),
                         glitch_two=np.nanmean(S.levels(g2)[0] == 2), gain12=np.nanmean(S.levels(g12)[0] == 2))
                rec = []
                for k, fx in faulty.items():
                    l, _ = S.levels(fx)
                    r[k] = np.nanmean(l == 2)
                    r[k + " | 저부하"] = np.nanmean(l[ll] == 2) if ll.any() else np.nan
                    rec.append(r[k])
                r["fault_recall"] = np.mean(rec)
                rows.append(r)
                print(f"s{seed} f{fold} {name}: FA {r['false_alarm']:.3f} fault {r['fault_recall']:.3f}", flush=True)
            clear_cache()
    d = pd.DataFrame(rows)
    d.to_csv(out, index=False)
    return d


if __name__ == "__main__":
    main()
