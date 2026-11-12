"""Evaluation of every model in models.catalogue() under three split protocols.

Protocols (splits are made on NORMAL segments before any window is cut):
  random   5-fold segment CV x 2 seeds   (all operating states appear in training)
  blocks   5 contiguous time blocks, each held out once (also used as streams)
  forward  first 80% of segments in time -> train, last 20% -> test (includes the low-load period)
Test sets per fold:
  normal   held-out normal segments (false alarms)
  real     the 21 segments of outlier_data.csv (recall on the provided abnormal records)
  faults   11 synthetic faults injected into held-out normal segments >= 16 rows (1:1 with clean copies)
  glitch   one wild current sample per held-out segment (nuisance: should NOT alarm)
Primary decision = Warning (fused score >= OOF q99 -> 1% false-alarm target); Attention = q95.
Writes results/raw/<protocol>_*.csv
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
from ..data import load_segments
from ..faults import FAULTS, glitch
from ..models import catalogue, FINAL, DETAIL, M2, M3, JOINT, CURRENT, VIBONLY
from ..detectors import MCD
from ..system import fill_short, diagnose, ewma
from ..detectors import clear_cache

PICK = [("진동 전체 증가", 1.5), ("진동 전체 증가", 2.0), ("충격 스파이크(16행당 1회)", 2), ("충격 스파이크(16행당 1회)", 3),
        ("하부 진동 극성 반전", 1), ("전류 오프셋 이동", 5), ("전류 오프셋 이동", 10), ("전류 진폭 증가", 1.2),
        ("전류 진폭 증가", 1.3), ("전류 파형 왜곡(2.5Hz 성분)", 0.05), ("전류 파형 왜곡(2.5Hz 성분)", 0.1)]
VIB_FAULTS = {"진동 전체 증가", "충격 스파이크(16행당 1회)", "하부 진동 극성 반전"}
RAMP = {"진동 전체 증가": (1.0, 2.0), "충격 스파이크(16행당 1회)": (0.0, 4.0), "전류 오프셋 이동": (0.0, 10.0),
        "전류 진폭 증가": (1.0, 1.3), "전류 파형 왜곡(2.5Hz 성분)": (0.0, 0.1)}
ONSET = 30
STREAM_MODELS = [FINAL, M2, M3, "M1 통합 모델 (9특징 MCD)", "B4' 형상+스펙트럼 (첨부안 원래 규칙: 둘 다 q99)"]


def low_load(seg):
    amp = (seg.x[:, 2].max() - seg.x[:, 2].min()) / 2
    return bool(15000 <= seg.row_index[0] < 19500 and abs(amp - 116) < 3)


def _auc(neg, pos):
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    s = np.r_[neg, pos]
    ok = ~np.isnan(s)
    if len(np.unique(y[ok])) < 2:
        return np.nan, np.nan
    return roc_auc_score(y[ok], s[ok]), average_precision_score(y[ok], s[ok])


def _runs(a):
    """number of alarm episodes (runs of consecutive True) and the longest run"""
    a = np.asarray(a, bool)
    if not a.any():
        return 0, 0
    starts = np.sum(a & ~np.r_[False, a[:-1]])
    longest, cur = 0, 0
    for v in a:
        cur = cur + 1 if v else 0
        longest = max(longest, cur)
    return int(starts), int(longest)


def stream_rules(S, train_segs):
    """single-strict + EWMA warning rule calibrated on the out-of-fold stream (1% target)"""
    order = np.argsort([s.t0 for s in train_segs])
    oof = S.oof[order]
    z_ref = S.z(oof[~np.isnan(oof)])
    strict = np.quantile(z_ref, 0.997)
    er = ewma(z_ref)
    cands = np.quantile(er, np.linspace(0.99, 0.9999, 60))
    e_th = next((c for c in cands if np.mean((z_ref > strict) | (er > c)) <= 0.01), cands[-1])
    return lambda z: (z > strict) | (ewma(z) > e_th)


def run_fold(protocol, fold, seed, train, test, O, models, rows, segrows, streamrows, do_stream=False):
    rng = np.random.default_rng(1000 * seed + fold)
    xs = [s.x for s in train]
    tx, tt = [s.x for s in test], [s.t0 for s in test]
    xo, to = [s.x for s in O], [s.t0 for s in O]
    long_idx = [i for i, s in enumerate(test) if s.n >= 16]
    clean = [tx[i] for i in long_idx]
    sd = np.concatenate([x[:, :2] for x in xs]).std(0)
    faulty = {f"{f} {v}": [FAULTS[f][2](x, v, sd, rng) for x in clean] for f, v in PICK}
    gl = [glitch(x, rng) for x in clean]
    for name, mk in models.items():
        t_start = time.time()
        S = mk().fit(xs, seed=seed)
        lv_n, s_n = S.levels(tx)
        lv_n, s_n = fill_short(lv_n, tt), fill_short(s_n, tt)
        lv_o, s_o = S.levels(xo)
        lv_o, s_o = fill_short(lv_o, to), fill_short(s_o, to)
        base = dict(protocol=protocol, fold=fold, seed=seed, model=name)
        fp, tn = int(np.sum(lv_n == 2)), int(np.sum(lv_n < 2))
        tp = int(np.sum(lv_o == 2))
        auc, ap = _auc(s_n, s_o)
        rows.append(dict(base, set="정상", n=len(tx), alarm=fp / len(tx), attention=np.mean(lv_n >= 1)))
        rows.append(dict(base, set="실제 이상", n=len(xo), tp=tp, fn=len(xo) - tp, fp=fp, tn=tn, auc=auc, ap=ap,
                         recall=tp / len(xo), attention=np.mean(lv_o >= 1)))
        lv_c, s_c = S.levels(clean)
        fp_c = np.nansum(lv_c == 2)
        n_c = np.sum(~np.isnan(lv_c))
        for k, fx in faulty.items():
            lv_f, s_f = S.levels(fx)
            ok = ~np.isnan(lv_f)
            tp_f = int(np.sum(lv_f[ok] == 2))
            a, p = _auc(s_c, s_f)
            rows.append(dict(base, set=k, family="진동" if k.rsplit(" ", 1)[0] in VIB_FAULTS else "전류",
                             n=int(ok.sum()), tp=tp_f, fn=int(ok.sum()) - tp_f, fp=int(fp_c), tn=int(n_c - fp_c),
                             auc=a, ap=p, recall=tp_f / max(ok.sum(), 1), attention=np.mean(lv_f[ok] >= 1)))
        lv_g, _ = S.levels(gl)
        rows.append(dict(base, set="측정 튐(1샘플)", n=int(np.sum(~np.isnan(lv_g))), alarm=np.nanmean(lv_g == 2)))
        print(f"  {protocol} f{fold} s{seed} {name}: {time.time() - t_start:.0f}s", flush=True)

        if name in DETAIL:
            vib_d, cur_d = VIBONLY().fit(xs), CURRENT().fit(xs)
            # feature contributions come from the MCD joint view (fitted separately when the joint view is a mixture)
            jd = S.det["joint"] if isinstance(S.det["joint"].m[24], MCD) else JOINT().fit(xs)
            VPn = S.view_p(tx)
            for i, s in enumerate(test):
                c = jd.window_contributions(s.x)
                segrows.append(dict(base, kind="정상", seg=s.seg_id, n=s.n, gap=s.gap_before, t0=s.t0, low_load=low_load(s),
                                    score=s_n[i], level=lv_n[i], p_joint=VPn[i, 0], p_current=VPn[i, 1],
                                    diag=diagnose(vib_d, cur_d, s.x) if lv_n[i] == 2 else "",
                                    **({f"c{j}": v for j, v in enumerate(c.mean(0))} if c is not None else {})))
            VPo = S.view_p(xo)
            for i, s in enumerate(O):
                c = jd.window_contributions(s.x)
                segrows.append(dict(base, kind="실제 이상", seg=s.seg_id, n=s.n, gap=s.gap_before, t0=s.t0, low_load=False,
                                    score=s_o[i], level=lv_o[i], p_joint=VPo[i, 0], p_current=VPo[i, 1],
                                    diag=diagnose(vib_d, cur_d, s.x),
                                    **({f"c{j}": v for j, v in enumerate(c.mean(0))} if c is not None else {})))
            for k, fx in faulty.items():
                VPf = S.view_p(fx)
                lv_f, _ = S.levels(fx, VPf)
                for i, x in enumerate(fx):
                    segrows.append(dict(base, kind=k, seg=test[long_idx[i]].seg_id, n=len(x), low_load=low_load(test[long_idx[i]]),
                                        level=lv_f[i], p_joint=VPf[i, 0], p_current=VPf[i, 1],
                                        diag=diagnose(vib_d, cur_d, x) if lv_f[i] == 2 else ""))
            # robustness / falsification checks on the final model
            checks = {"이득 ×1.1 (모든 센서)": [x * 1.1 for x in clean], "이득 ×1.2 (모든 센서)": [x * 1.2 for x in clean],
                      "조각 내 시간 순서 섞기 (정상)": [x[rng.permutation(len(x))] for x in clean],
                      "순환 이동 (정상)": [np.roll(x, len(x) // 2, axis=0) for x in clean]}
            for k, xx in checks.items():
                lv_k, _ = S.levels(xx)
                rows.append(dict(base, set="점검: " + k, n=len(xx), alarm=np.nanmean(lv_k == 2)))
            sh = [x[rng.permutation(len(x))] for x in xo]
            lv_k, _ = S.levels(sh)
            rows.append(dict(base, set="점검: 조각 내 시간 순서 섞기 (실제 이상)", n=len(xo),
                             recall=np.nanmean(fill_short(lv_k, to) == 2)))

        if do_stream and name in STREAM_MODELS:
            order = np.argsort(tt)
            sc_all = S.score(tx)
            sx = [tx[i] for i in order if not np.isnan(sc_all[i])]
            st = [tt[i] for i in order if not np.isnan(sc_all[i])]
            hours = (max(st) - min(st)).total_seconds() / 3600
            lv_s, s_s = S.levels(sx)
            single = lv_s == 2
            sb = dict(base, hours=hours, n_seg=len(sx))
            ep, mx = _runs(single)
            streamrows.append(dict(sb, rule="단일 조각 경고", scenario="정상 스트림", alarm=single.mean(), episodes=ep,
                                   episodes_per_hour=ep / hours, max_run=mx, attention=np.mean(lv_s >= 1)))
            rule = stream_rules(S, train) if S.rule == "fused" else None
            if rule is not None:
                a = rule(S.z(s_s))
                ep, mx = _runs(a)
                streamrows.append(dict(sb, rule="단일(엄격)+EWMA 경고", scenario="정상 스트림", alarm=a.mean(), episodes=ep,
                                       episodes_per_hour=ep / hours, max_run=mx))
            for rep in range(2):
                r2 = np.random.default_rng(77 + 10 * fold + rep)
                for fname, (lo, hi) in RAMP.items():
                    n = len(sx)
                    sev = [lo if i < ONSET else lo + (hi - lo) * (i - ONSET + 1) / (n - ONSET) for i in range(n)]
                    fx = [x if i < ONSET else FAULTS[fname][2](x, v, sd, r2) for i, (x, v) in enumerate(zip(sx, sev))]
                    lv_r, s_r = S.levels(fx)
                    alarms = {"단일 조각 경고": lv_r == 2}
                    if rule is not None:
                        alarms["단일(엄격)+EWMA 경고"] = rule(S.z(s_r))
                    for rn, a in alarms.items():
                        hit = np.where(a[ONSET:])[0]
                        streamrows.append(dict(sb, rule=rn, scenario=fname, rep=rep,
                                               first_alarm_severity=sev[ONSET + hit[0]] if len(hit) else np.nan,
                                               false_before=int(a[:ONSET].sum())))
                burst = [xo[i] for i in r2.choice(len(xo), 10, replace=False) if len(xo[i]) >= 8][:8]
                lv_b, s_b = S.levels(sx[:ONSET] + burst)
                alarms = {"단일 조각 경고": lv_b == 2}
                if rule is not None:
                    alarms["단일(엄격)+EWMA 경고"] = rule(S.z(s_b))
                for rn, a in alarms.items():
                    hit = np.where(a[ONSET:])[0]
                    streamrows.append(dict(sb, rule=rn, scenario="실제 이상 갑자기 등장", rep=rep,
                                           delay_segments=hit[0] + 1 if len(hit) else np.nan, false_before=int(a[:ONSET].sum())))
    clear_cache()


def main(protocol, out_dir="results/raw", data_dir="data", only=None, tag=""):
    os.makedirs(out_dir, exist_ok=True)
    N = load_segments(os.path.join(data_dir, "press_data_normal.csv"), 0)
    O = load_segments(os.path.join(data_dir, "outlier_data.csv"), 1)
    models = catalogue()
    if only:
        models = {k: v for k, v in models.items() if k in only}
    rows, segrows, streamrows = [], [], []
    if protocol == "random":
        for seed in [0, 1]:
            for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=seed).split(np.arange(len(N)))):
                run_fold("random", fold, seed, [N[i] for i in tr], [N[i] for i in te], O, models, rows, segrows, streamrows)
    elif protocol == "blocks":
        order = np.argsort([s.t0 for s in N])
        for fold, blk in enumerate(np.array_split(order, 5)):
            b = set(blk)
            run_fold("blocks", fold, 0, [N[i] for i in order if i not in b], [N[i] for i in blk], O, models,
                     rows, segrows, streamrows, do_stream=True)
    elif protocol == "forward":
        order = np.argsort([s.t0 for s in N])
        cut = int(0.8 * len(order))
        run_fold("forward", 0, 0, [N[i] for i in order[:cut]], [N[i] for i in order[cut:]], O, models, rows, segrows, streamrows)
    pd.DataFrame(rows).to_csv(f"{out_dir}/{protocol}{tag}_metrics.csv", index=False)
    pd.DataFrame(segrows).to_csv(f"{out_dir}/{protocol}{tag}_segments.csv", index=False)
    if streamrows:
        pd.DataFrame(streamrows).to_csv(f"{out_dir}/{protocol}{tag}_stream.csv", index=False)


if __name__ == "__main__":
    only = sys.argv[2].split("|") if len(sys.argv) > 2 else None
    main(sys.argv[1], only=only, tag=sys.argv[3] if len(sys.argv) > 3 else "")
