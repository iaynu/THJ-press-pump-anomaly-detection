"""Calibrated anomaly-scoring system built from one or more detectors ('views').

Calibration (the same for every model, so comparisons are fair):
  1. Out-of-fold (OOF) scores: the training (normal) segments are split into 5 folds; each fold is
     scored by a detector that never saw it.
  2. Each view's score becomes a 'normal percentile' p = share of OOF normal scores below it,
     i.e. 1 - p is how rare the segment is among normal operation (a calibrated rarity, not a
     fault probability).
  3. Views are fused: 'max' (alarm if ANY view finds it rare) or 'min' (alarm only if ALL views do).
  4. Thresholds are quantiles of the fused OOF scores: attention q95, warning q99.
     The 'and_each' rule reproduces the attached report's original Warning: every view >= its own q99.
  5. Optional robust calibration (trim_z): before steps 2-4, training segments whose log OOF score is more than
     trim_z robust standard deviations (median/MAD) above the median in any view are left out, so that faulty
     segments hidden in the 'normal' label cannot drag the thresholds up.
Segments too short to judge take the decision of the nearest judged segment in time.
"""
import numpy as np
from scipy.stats import norm, chi2
from sklearn.model_selection import KFold


def _pct(ref_sorted, s):
    p = np.searchsorted(ref_sorted, s, side="right") / (len(ref_sorted) + 1)
    return np.where(np.isnan(s), np.nan, p)


def oof_scores(factory, xs, k=5, seed=0):
    out = np.full(len(xs), np.nan)
    for tr, te in KFold(k, shuffle=True, random_state=seed).split(np.arange(len(xs))):
        det = factory().fit([xs[i] for i in tr])
        out[te] = det.seg_scores([xs[i] for i in te])
    return out


class System:
    def __init__(self, views, fusion="max", q_att=0.95, q_warn=0.99, rule="fused", trim_z=None):
        self.views, self.fusion, self.q_att, self.q_warn, self.rule = views, fusion, q_att, q_warn, rule
        self.trim_z = trim_z   # robust calibration: drop OOF segments with robust log-score z > trim_z before setting thresholds

    def _fuse(self, P):
        P = np.column_stack(P)
        with np.errstate(all="ignore"):
            f = P.max(1) if self.fusion == "max" else P.min(1)
        return np.where(np.isnan(P).any(1), np.nan, f)

    def fit(self, xs, seed=0):
        raw = {name: oof_scores(fac, xs, seed=seed) for name, fac in self.views.items()}
        self.det = {name: fac().fit(xs) for name, fac in self.views.items()}
        return self.calibrate(raw)

    def calibrate(self, raw):
        """set reference distributions and thresholds from raw out-of-fold scores {view: scores}"""
        self.raw_oof, self.ref, oof_p = raw, {}, []
        keep = np.ones(len(next(iter(raw.values()))), bool)
        if self.trim_z is not None:   # suspected label noise: segments far outside the robust bulk of any view's scores
            for o in raw.values():
                lo = np.log(np.maximum(o - np.nanmin(o) + 1e-9, 1e-9))
                med = np.nanmedian(lo)
                mad = 1.4826 * np.nanmedian(np.abs(lo - med)) + 1e-12
                keep &= ~((lo - med) / mad > self.trim_z)
        self.keep = keep
        self.trimmed_share = 1 - keep.mean()
        for name, o in raw.items():
            ok = keep & ~np.isnan(o)
            self.ref[name] = np.sort(o[ok])
            oof_p.append(np.where(keep, _pct(self.ref[name], o), np.nan))
        self.oof_view_p = np.column_stack(oof_p)
        self.oof = self._fuse(oof_p)
        ok = self.oof[~np.isnan(self.oof)]
        self.t_att, self.t_warn = np.quantile(ok, self.q_att), np.quantile(ok, self.q_warn)
        self.t_view_warn = np.nanquantile(self.oof_view_p, self.q_warn, axis=0)
        self.t_view_att = np.nanquantile(self.oof_view_p, self.q_att, axis=0)
        self.ref_fused = np.sort(ok)
        return self

    def view_p(self, xs):
        return np.column_stack([_pct(self.ref[n], self.det[n].seg_scores(xs)) for n in self.views])

    def score(self, xs):
        """fused normal-percentile score per segment (nan = too short)"""
        return self._fuse(list(self.view_p(xs).T))

    def levels(self, xs, VP=None):
        """0 normal / 1 attention / 2 warning per segment (nan if too short)"""
        VP = self.view_p(xs) if VP is None else VP
        s = self._fuse(list(VP.T))
        if self.rule == "and_each":     # attached report: attention = any view >= q95, warning = all views >= q99
            att = (VP >= self.t_view_att).any(1)
            warn = (VP >= self.t_view_warn).all(1)
        else:
            att, warn = s >= self.t_att, s >= self.t_warn
        lv = np.where(warn, 2, np.where(att, 1, 0)).astype(float)
        return np.where(np.isnan(s), np.nan, lv), s

    def z(self, s):
        """fused score -> normal quantile (for the EWMA trend rule)"""
        p = (np.searchsorted(self.ref_fused, s, side="right") + 0.5) / (len(self.ref_fused) + 1)
        return norm.ppf(np.clip(p, 1e-6, 1 - 1e-6))


def fill_short(values, times):
    """segments without a value take the value of the nearest segment in time that has one"""
    v = np.array(values, dtype=float)
    t = np.array([np.datetime64(x) for x in times]).astype("datetime64[ms]").astype(np.int64)
    have = np.where(~np.isnan(v))[0]
    for i in np.where(np.isnan(v))[0]:
        if len(have):
            v[i] = v[have[np.argmin(np.abs(t[have] - t[i]))]]
    return v


def diagnose(vib_det, cur_det, x, df_v=6, df_c=3, strong=1e-3):
    """vibration-type / current-type / both, from separate vibration-only and current-only MCD models"""
    W = vib_det.route(x)
    if W is None:
        return "판정 불가"
    pv = chi2.logsf(vib_det.m[W].score(vib_det._F(x, W)), df_v).mean()
    pc = chi2.logsf(cur_det.m[W].score(cur_det._F(x, W)), df_c).mean()
    if pv < np.log(strong) and pc < np.log(strong):
        return "복합형"
    return "진동형" if pv < pc else "전류형"


def ewma(z, lam=0.2):
    out, e = [], 0.0
    for v in z:
        e = lam * v + (1 - lam) * e
        out.append(e)
    return np.array(out)
