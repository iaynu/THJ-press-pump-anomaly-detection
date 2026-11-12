"""Window features (9 per window).

Vibration (6): RMS and peak of upper (AI0) and lower (AI1), RMS of (upper+lower) and (upper-lower).
Current (3): a 0.6 Hz sine + constant least-squares fit per window gives the centre line (DC offset),
the amplitude, and the residual ratio (waveform distortion). One or two wild samples are repaired
before the fit (single-sample glitch repair).

Why these: amplitude growth and impacts are the classic pump-wear signals (RMS, peak); the sum/diff
pair sees the two vibration sensors moving together vs against each other; the current is a very
regular 0.59-0.62 Hz sine in normal data, so its centre, size and distortion are the natural summaries.
The upper-lower correlation is deliberately NOT used: it flips sign in the normal low-load state and
caused false alarms. Note: 10 Hz sampling means only 0-5 Hz is observable, so nothing here is a motor
electrical fault frequency.
"""
from functools import lru_cache
import numpy as np

FEATURES = ["상부 RMS", "하부 RMS", "상부 피크", "하부 피크", "합 RMS", "차 RMS",
            "전류 중심선", "전류 진폭", "전류 왜곡"]
VIB = [0, 1, 2, 3, 4, 5]
CUR = [6, 7, 8]
GRID = np.linspace(0.55, 0.65, 11)


def cut(x, W, stride=4):
    if len(x) < W:
        return np.zeros((0, W, x.shape[1]))
    return np.stack([x[a:a + W] for a in range(0, len(x) - W + 1, stride)])


@lru_cache(maxsize=None)
def _design(W):
    t = np.arange(W) * 0.1
    Ds = [np.column_stack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t), np.ones(W)]) for f in GRID]
    return Ds, [np.linalg.pinv(D) for D in Ds]


def _fit(c):
    Ds, Ps = _design(c.shape[1])
    best_res = best = None
    bestk = np.zeros(len(c), int)
    for k, (D, P) in enumerate(zip(Ds, Ps)):
        coef = c @ P.T
        res = ((c - coef @ D.T) ** 2).mean(1)
        if best_res is None:
            best_res, best = res, coef
        else:
            m = res < best_res
            best_res, best = np.where(m, res, best_res), np.where(m[:, None], coef, best)
            bestk = np.where(m, k, bestk)
    fitted = np.einsum("nk,ntk->nt", best, np.stack([Ds[k] for k in bestk])) if len(c) else c
    return best, best_res, fitted


def loo_repair(c, tau=0.3, rounds=2, events=None):
    """Leave-one-out glitch repair. For every window and every grid frequency, the closed form of the leave-one-out
    residual sum of squares (RSS_(j) = RSS - e_j^2 / (1 - h_jj)) says how much of the misfit ONE sample explains.
    If dropping the best single sample leaves less than tau of the RSS, that sample is a glitch and is replaced by the
    value predicted without it. Up to `rounds` samples per window. A distorted waveform has no such single culprit."""
    c = c.copy()
    n, W = c.shape
    if n == 0:
        return c
    Ds, Ps = _design(W)
    ar = np.arange(n)
    for _ in range(rounds):
        best_rss = np.full(n, np.inf)
        cand = np.full(n, np.inf)
        cj = np.zeros(n, int)
        cval = np.zeros(n)
        for D, P in zip(Ds, Ps):
            e = c - (c @ P.T) @ D.T
            rss = (e ** 2).sum(1)
            lev = np.maximum(1 - np.einsum("jk,kj->j", D, P), 1e-3)
            rss_j = rss[:, None] - e ** 2 / lev[None, :]
            j = rss_j.argmin(1)
            r = rss_j[ar, j]
            m = r < cand
            cand, cj = np.where(m, r, cand), np.where(m, j, cj)
            cval = np.where(m, c[ar, j] - e[ar, j] / lev[j], cval)
            best_rss = np.minimum(best_rss, rss)
        hit = (cand < tau * best_rss) & (best_rss > 1e-9)
        if not hit.any():
            break
        c[ar[hit], cj[hit]] = cval[hit]
        if events is not None:
            events.extend(zip(ar[hit].tolist(), cj[hit].tolist(), cval[hit].tolist()))
    return c


def segment_glitch_repair(x, W, tau=0.3, share=0.5):
    """Window-level repair is inconsistent: the same wild sample is repaired in some overlapping windows and not in
    others. Here every stride-1 window of the segment votes (leave-one-out) on which sample is the glitch; a sample
    flagged in at least `share` of the windows that contain it is replaced, once, by the median predicted value."""
    n = len(x)
    if n < W:
        return x
    X = np.stack([x[a:a + W, 2] for a in range(n - W + 1)])
    ev = []
    loo_repair(X, tau, events=ev)
    if not ev:
        return x
    votes, vals = np.zeros(n), [[] for _ in range(n)]
    for w, j, v in ev:
        votes[w + j] += 1
        vals[w + j].append(v)
    idx = np.arange(n)
    cover = np.minimum(idx, n - W) - np.maximum(0, idx - W + 1) + 1
    bad = np.where(votes >= np.maximum(1, share * cover))[0]
    if len(bad) == 0:
        return x
    x = x.copy()
    for i in bad:
        x[i, 2] = np.median(vals[i])
    return x


def sine_fit(c, repair=True, mode="thr", tau=0.3):
    """mode: "thr" = replace up to 2 samples with |residual| > 4 sigma (default);
    "loo" = leave-one-out repair only;  "both" = leave-one-out first, then the threshold rule"""
    if mode in ("loo", "both") and len(c):
        c = loo_repair(c, tau)
        repair = mode == "both"
    best, res, fitted = _fit(c)
    if repair and len(c):
        r = c - fitted
        amp0 = np.hypot(best[:, 0], best[:, 1])[:, None]
        s = np.maximum(1.4826 * np.median(np.abs(r), 1, keepdims=True), 0.02 * amp0)
        big = (np.abs(r) > 4 * s) & (np.argsort(-np.abs(r), 1).argsort(1) < 2)
        if big.any():
            best, res, _ = _fit(np.where(big, fitted, c))
    amp = np.hypot(best[:, 0], best[:, 1])
    return best[:, 2], amp, np.sqrt(res) / np.maximum(amp, 1e-9)


def _rms(z):
    return np.sqrt((z ** 2).mean(1))


def window_features(X, repair="thr", tau=0.3):
    """X: (n, W, 3) -> (n, 9)"""
    if len(X) == 0:
        return np.zeros((0, 9))
    a, b = X[:, :, 0], X[:, :, 1]
    off, amp, res = sine_fit(X[:, :, 2], mode=repair, tau=tau)
    return np.column_stack([_rms(a), _rms(b), np.abs(a).max(1), np.abs(b).max(1),
                            _rms(a + b), _rms(a - b), off, amp, res])
