"""Back end of the monitoring dashboard (app.py): replay a stream of segments through the final system and
turn the per-segment result into a machine-state decision (RUN / WATCH / WARN / STOP request).

Honest demo set-up: the dashboard model is fitted on the FIRST 80% (in time) of the normal segments only and the
demo streams are built from the LAST 20% (never seen in training), the 21 real anomalous segments and synthetic faults
injected into those held-out segments. The model shipped for scoring new files (models/final_system.pkl) uses all
normal data; this one exists only so that the demo does not score segments it was trained on.

Decision logic per completed segment (one step = one segment):
  level      0 normal / 1 attention / 2 warning   (final system: q95 / q99 of out-of-fold normal scores)
  trend      single extreme segment (z > q99.7) OR EWMA of the normal-quantile score above its calibrated limit
  STOP       `stop_k` warning segments in a row   -> stop request (a simulated interlock signal, not a plant control)
             OR trend alarm on `stop_k` segments in a row
  Segments too short to judge (< 8 rows) keep the previous state and do not count.
  python -m kamp.dashboard_core build      fit the dashboard model -> models/dashboard_system.pkl
  python -m kamp.dashboard_core simulate   stop-rule check on the held-out stream and the scenarios
"""
import os
import pickle
import sys
import warnings
from dataclasses import dataclass, field
import numpy as np
warnings.filterwarnings("ignore")
from .data import load_segments
from .faults import FAULTS, scale, shift, cur_amp, spikes, harmonic
from .features import FEATURES
from .models import final_system, VIBONLY, CURRENT
from .system import diagnose
from .evaluate import _ewma, glitch

LEVEL_NAME = {0: "정상", 1: "주의", 2: "경고"}
HOLDOUT_SHARE = 0.2
STOP_K = 2
PATH = "models/dashboard_system.pkl"


# --------------------------------------------------------------------------------------------- model
def build(data_dir="data", out=PATH):
    N = load_segments(os.path.join(data_dir, "press_data_normal.csv"), 0)
    order = np.argsort([s.t0 for s in N])
    cut = int((1 - HOLDOUT_SHARE) * len(order))
    tr, ho = order[:cut], order[cut:]
    xs = [N[i].x for i in tr]
    S = final_system().fit(xs, seed=0)
    oof = S.oof[np.argsort([N[i].t0 for i in tr])]          # out-of-fold scores in time order
    z_ref = S.z(oof[~np.isnan(oof)])
    strict = float(np.quantile(z_ref, 0.997))
    er = _ewma(z_ref)
    cands = np.quantile(er, np.linspace(0.99, 0.9999, 60))
    e_th = float(next((c for c in cands if np.mean((z_ref > strict) | (er > c)) <= 0.01), cands[-1]))
    bundle = dict(system=S, vib_diag=VIBONLY().fit(xs), cur_diag=CURRENT().fit(xs),
                  trend=dict(strict_z=strict, ewma_z=e_th, lam=0.2), holdout=[int(i) for i in ho], train=[int(i) for i in tr],
                  sd=np.concatenate([x[:, :2] for x in xs]).std(0), n_train=len(tr))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(bundle, f)
    return bundle


def load(path=PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------------------------- scenarios
def _ramp(i, onset, length, lo, hi):
    """severity: lo before onset, rising linearly from lo at onset to hi after `length` segments"""
    return lo if i < onset else lo + (hi - lo) * min(1.0, (i - onset + 1) / length)


SCENARIOS = {
    "정상 운전 (미학습 구간)": "고장 없음. 시스템이 불필요하게 정지시키지 않는지 확인합니다.",
    "실제 이상 기록": "정상 구간 뒤에 실제 이상 기록(outlier_data) 21조각이 이어집니다.",
    "진동 점진 증가 (마모)": "상·하부 진동이 1.0배에서 2.0배까지 서서히 커집니다. (주입 고장)",
    "전류 중심선 점진 이동": "전류 중심선이 0에서 +40까지 서서히 이동합니다. (주입 고장)",
    "전류 진폭 점진 증가": "전류 진폭이 1.0배에서 1.5배까지 서서히 커집니다. (주입 고장)",
    "충격 스파이크 돌발": "어느 시점부터 진동에 3σ 충격이 반복됩니다. (주입 고장)",
    "하부 진동 극성 반전 (배선 오류)": "어느 시점부터 하부 진동 부호가 뒤집힙니다. (주입 고장)",
    "전류 1샘플 튐 (센서 노이즈)": "일부 조각에 전류 한 샘플만 튑니다. 정지하면 안 되는 상황입니다.",
}


@dataclass
class Step:
    i: int
    x: np.ndarray
    t0: object
    truth: str            # "정상" / "이상(실제)" / "이상(주입)" / "노이즈(주입)"
    severity: float = np.nan


def make_stream(name, N, O, B, seed=0, length=60, onset=20):
    """list of Step for a scenario. N: all normal segments, O: real anomalous segments, B: dashboard bundle"""
    rng = np.random.default_rng(seed)
    ho = sorted(B["holdout"], key=lambda i: N[i].t0)
    base = [N[i] for i in ho]
    sd = B["sd"]
    if name.startswith("정상 운전"):
        return [Step(i, s.x, s.t0, "정상") for i, s in enumerate(base)]
    if name == "실제 이상 기록":
        out = [Step(i, s.x, s.t0, "정상") for i, s in enumerate(base[:onset])]
        return out + [Step(onset + j, s.x, s.t0, "이상(실제)") for j, s in enumerate(O)]
    base = base[:length]
    out = []
    for i, s in enumerate(base):
        x, truth, sev = s.x, "정상", np.nan
        if i >= onset:
            truth = "이상(주입)"
            if name.startswith("진동 점진"):
                sev = _ramp(i, onset, 30, 1.1, 2.0)
                x = scale(x, sev, [0, 1])
            elif name.startswith("전류 중심선"):
                sev = _ramp(i, onset, 30, 4.0, 40.0)
                x = shift(x, sev)
            elif name.startswith("전류 진폭"):
                sev = _ramp(i, onset, 30, 1.05, 1.5)
                x = cur_amp(x, sev)
            elif name.startswith("충격"):
                sev = 3.0
                x = spikes(x, sev, sd, rng)
            elif name.startswith("하부 진동 극성"):
                sev = -1.0
                x = scale(x, -1, [1])
            elif name.startswith("전류 1샘플"):
                truth = "노이즈(주입)"
                if rng.random() < 0.5:
                    x = glitch(x, rng, 0.5)
        out.append(Step(i, x, s.t0, truth, sev))
    return out


# --------------------------------------------------------------------------------------------- monitor
@dataclass
class Result:
    i: int
    judged: bool
    level: int
    pct: float            # fused normal percentile (1 - pct = how rare among normal segments)
    z: float
    ewma: float
    trend: bool
    consec_warn: int
    consec_trend: int
    stop: bool
    reason: str = ""
    diag: str = ""
    contrib: np.ndarray = None          # joint-view feature contributions (9)
    p_joint: float = np.nan
    p_current: float = np.nan


class Monitor:
    """stateful replay of the decision logic; call .step(x) once per completed segment"""
    def __init__(self, B, stop_k=STOP_K, auto_stop=True):
        self.B, self.S, self.stop_k, self.auto_stop = B, B["system"], stop_k, auto_stop
        self.reset()

    def reset(self):
        self.e, self.cw, self.ct, self.level, self.stopped, self.n = 0.0, 0, 0, 0, False, 0

    def acknowledge(self):
        """operator clears the stop request (machine inspected)"""
        self.stopped, self.cw, self.ct = False, 0, 0

    def step(self, x):
        S, T = self.S, self.B["trend"]
        i, self.n = self.n, self.n + 1
        VP = S.view_p([x])
        lv, sc = S.levels([x], VP)
        if np.isnan(lv[0]):             # too short to judge: hold the previous state
            return Result(i, False, self.level, np.nan, np.nan, self.e, False, self.cw, self.ct, self.stopped)
        lv, sc = int(lv[0]), float(sc[0])
        z = float(S.z(np.array([sc]))[0])
        self.e = T["lam"] * z + (1 - T["lam"]) * self.e
        trend = z > T["strict_z"] or self.e > T["ewma_z"]
        self.cw = self.cw + 1 if lv == 2 else 0
        self.ct = self.ct + 1 if trend else 0
        self.level = lv
        reason = ""
        if self.auto_stop and not self.stopped and (self.cw >= self.stop_k or self.ct >= self.stop_k):
            self.stopped = True
            reason = f"경고 {self.cw}조각 연속" if self.cw >= self.stop_k else f"추세 경보 {self.ct}조각 연속"
        r = Result(i, True, lv, sc, z, self.e, trend, self.cw, self.ct, self.stopped, reason, p_joint=float(VP[0, 0]), p_current=float(VP[0, 1]))
        if lv >= 1 or trend:
            r.diag = diagnose(self.B["vib_diag"], self.B["cur_diag"], x)
            c = S.det["joint"].window_contributions(x)
            r.contrib = None if c is None else c.mean(0)
        return r


def run_stream(B, steps, stop_k=STOP_K):
    M = Monitor(B, stop_k)
    return [M.step(s.x) for s in steps]


def summarize(steps, results):
    """per-run numbers shown in the dashboard: first warning / stop, delay after the fault begins, stops before it"""
    truth = np.array([s.truth for s in steps])
    bad = np.array([t.startswith("이상") for t in truth])
    stop = np.array([r.stop for r in results])
    warn = np.array([r.level == 2 and r.judged for r in results])
    first_bad = int(np.argmax(bad)) if bad.any() else None
    fs = int(np.argmax(stop)) if stop.any() else None
    fw = int(np.argmax(warn[first_bad:])) + first_bad if first_bad is not None and warn[first_bad:].any() else None
    # stop episodes are counted until the first stop (after a stop the machine is halted in the demo)
    pre = slice(0, first_bad if first_bad is not None else len(steps))
    return dict(n=len(steps), fault_start=first_bad, first_stop=fs,
                stop_delay=None if fs is None or first_bad is None or fs < first_bad else fs - first_bad,
                false_stop=bool(stop[pre].any()), first_warning_after_onset=fw,
                warnings_in_normal=int(warn[pre].sum()), normal_steps=int((~bad).sum()))


def simulate(path=PATH, data_dir="data"):
    """check the stop rule: false stops on normal streams and delay on every scenario, for stop_k in 1..3"""
    import pandas as pd
    B = load(path)
    N = load_segments(os.path.join(data_dir, "press_data_normal.csv"), 0)
    O = load_segments(os.path.join(data_dir, "outlier_data.csv"), 1)
    rows = []
    for k in (1, 2, 3):
        for name in SCENARIOS:
            for seed in range(5 if name not in ("정상 운전 (미학습 구간)", "실제 이상 기록") else 1):
                steps = make_stream(name, N, O, B, seed)
                res = run_stream(B, steps, k)
                m = summarize(steps, res)
                rows.append(dict(stop_k=k, scenario=name, seed=seed, **m))
    d = pd.DataFrame(rows)
    os.makedirs("results/raw", exist_ok=True)
    d.to_csv("results/raw/dashboard_stop_rule.csv", index=False)
    g = d.groupby(["scenario", "stop_k"]).agg(false_stop=("false_stop", "mean"), delay=("stop_delay", "median"),
                                              missed=("first_stop", lambda s: s.isna().mean()),
                                              warnings_before=("warnings_in_normal", "mean"))
    print(g.round(2).to_string())
    return d


if __name__ == "__main__":
    {"build": lambda: (build(), print("saved", PATH)), "simulate": simulate}[sys.argv[1]]()
