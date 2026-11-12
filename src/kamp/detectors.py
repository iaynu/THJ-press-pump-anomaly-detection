"""Segment-level detectors. Every detector is trained on NORMAL segments only and returns one
anomaly score per segment (higher = more unusual; nan = segment too short to judge).

Our detectors (features.py, routed by segment length: >=24 rows -> 24-row windows,
16-23 -> 16, 8-15 -> 8; segment score = mean of its window scores):
  MCD            robust Mahalanobis distance (Minimum Covariance Determinant)
  GMM            Gaussian mixture, one component per operating state (negative log-likelihood)
  IForest        Isolation Forest                      (comparison)
  KNN            mean distance to 5 nearest normal windows (comparison)
  RMSRule        max z-score of upper/lower vibration RMS (simple rule baseline)
Other detectors:
  VibAE          dense autoencoder on raw 16-row vibration windows (deep comparison)
  WaveShape      per-channel z-normalised 10-row waveform, shrinkage (Ledoit-Wolf) Mahalanobis
  SpecShape      per-channel 0-5 Hz power-spectrum SHAPE (demean, Hann, drop DC, sum to 1), kNN
  (WaveShape + SpecShape = the structure proposed in the attached review report; top-2 mean.)
"""
import numpy as np
from sklearn.covariance import MinCovDet, LedoitWolf
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from .features import cut, window_features, segment_glitch_repair

ROUTES = ((24, 24), (16, 16), (8, 8))
_FCACHE = {}


def _feats(x, W, stride=4, repair="thr", tau=0.3):
    key = (id(x), W, stride, repair, tau)
    if key not in _FCACHE:
        if repair.startswith("seg"):      # "seg_thr" / "seg_both": consensus repair of the whole segment first
            xr, repair = segment_glitch_repair(x, W, tau), repair.split("_")[1]
        else:
            xr = x
        _FCACHE[key] = (x, window_features(cut(xr, W, stride), repair, tau))   # keep x alive so id() is never reused
    return _FCACHE[key][1]


def clear_cache():
    _FCACHE.clear()


class _Std:
    scaler = "std"      # "std": mean / standard deviation;  "robust": median / (IQR / 1.349)  (the Colab notebook's RobustScaler)

    def _std_fit(self, F):
        if self.scaler == "robust":
            q1, q3 = np.percentile(F, [25, 75], axis=0)
            self.mu_, self.sd_ = np.median(F, 0), (q3 - q1) / 1.349 + 1e-9
        else:
            self.mu_, self.sd_ = F.mean(0), F.std(0) + 1e-9
        return (F - self.mu_) / self.sd_

    def _std(self, F):
        return (F - self.mu_) / self.sd_


class MCD(_Std):
    def __init__(self, support=0.9, scaler="std"):
        self.support, self.scaler = support, scaler

    def fit(self, F):
        Z = self._std_fit(F)
        m = MinCovDet(random_state=0, support_fraction=self.support).fit(Z)
        self.loc_, self.P_ = m.location_, np.linalg.inv(m.covariance_ + 1e-6 * np.eye(Z.shape[1]))
        return self

    def _z(self, F):
        return self._std(F) - self.loc_

    def score(self, F):
        Z = self._z(F)
        return np.einsum("ij,jk,ik->i", Z, self.P_, Z)

    def contributions(self, F):
        Z = self._z(F)
        return Z * (Z @ self.P_.T)


class GMM(_Std):
    """Gaussian mixture (one component per operating state); score = negative log-likelihood"""
    def __init__(self, k=3):
        self.k = k

    def fit(self, F):
        self.m = GaussianMixture(self.k, covariance_type="full", random_state=0, reg_covar=1e-4).fit(self._std_fit(F))
        return self

    def score(self, F):
        return -self.m.score_samples(self._std(F))


class IForest(_Std):
    def fit(self, F):
        self.m = IsolationForest(n_estimators=200, random_state=0).fit(self._std_fit(F))
        return self

    def score(self, F):
        return -self.m.score_samples(self._std(F))


class KNN(_Std):
    def __init__(self, k=5, max_ref=6000):
        self.k, self.max_ref = k, max_ref

    def fit(self, F):
        Z = self._std_fit(F)
        if len(Z) > self.max_ref:
            Z = Z[np.random.default_rng(0).choice(len(Z), self.max_ref, replace=False)]
        self.nn = NearestNeighbors(n_neighbors=self.k).fit(Z)
        return self

    def score(self, F):
        return self.nn.kneighbors(self._std(F))[0].mean(1)


class RMSRule(_Std):
    """baseline rule: how far the upper or lower vibration RMS is above its normal level"""
    def fit(self, F):
        self._std_fit(F[:, :2])
        return self

    def score(self, F):
        return self._std(F[:, :2]).max(1)


class FeatureDetector:
    """agg: how window scores become the segment score: "mean" (ours), "max" or "top2" (mean of the two largest; the notebook's options)
    half_overlap: windows step by W/2 instead of 4 rows (the notebook's 50% overlap)
    abs_distortion: current distortion as absolute residual RMS instead of residual / amplitude (the notebook's definition)
    repair / tau: single-sample glitch repair of the current ("thr" default, "loo", "both"; see features.loo_repair)"""
    def __init__(self, model, cols=None, routes=ROUTES, name="", agg="mean", half_overlap=False, abs_distortion=False,
                 repair="thr", tau=0.3):
        self.model, self.cols, self.routes, self.name = model, cols, routes, name
        self.agg, self.half_overlap, self.abs_distortion = agg, half_overlap, abs_distortion
        self.repair, self.tau = repair, tau

    def route(self, x):
        return next((W for lo, W in self.routes if len(x) >= lo), None)

    def _F(self, x, W):
        F = _feats(x, W, W // 2 if self.half_overlap else 4, self.repair, self.tau)
        if self.abs_distortion:
            F = F.copy()
            F[:, 8] = F[:, 7] * F[:, 8]
        return F if self.cols is None else F[:, self.cols]

    def _agg(self, s):
        if self.agg == "max":
            return s.max()
        if self.agg == "top2":
            return np.sort(s)[-2:].mean()
        return s.mean()

    def fit(self, xs):
        self.m = {}
        for _, W in self.routes:
            F = np.vstack([self._F(x, W) for x in xs if len(x) >= W])
            self.m[W] = self.model().fit(F)
        return self

    def seg_scores(self, xs):
        out = np.full(len(xs), np.nan)
        for i, x in enumerate(xs):
            W = self.route(x)
            if W is not None:
                out[i] = self._agg(self.m[W].score(self._F(x, W)))
        return out

    def window_contributions(self, x):
        W = self.route(x)
        return None if W is None else self.m[W].contributions(self._F(x, W))


class _RawWindowDetector:
    """detectors that work on raw windows of one length; segment score = mean or top-2 mean"""
    W, stride, agg = 16, 4, "mean"

    def windows(self, x):
        return cut(x, self.W, self.stride)

    def seg_scores(self, xs):
        out = np.full(len(xs), np.nan)
        for i, x in enumerate(xs):
            X = self.windows(x)
            if len(X):
                s = np.sort(self.score_windows(X))[::-1]
                out[i] = s[:2].mean() if self.agg == "top2" else s.mean()
        return out

    def fit(self, xs):
        X = np.concatenate([self.windows(x) for x in xs if len(x) >= self.W])
        self.fit_windows(X)
        return self


class VibAE(_RawWindowDetector):
    W, stride, agg = 16, 4, "mean"

    def __init__(self, seed=0):
        self.seed = seed

    def _prep(self, X):
        return ((X[:, :, :2] - self.mu_) / self.sd_).reshape(len(X), -1)

    def fit_windows(self, X):
        self.mu_ = X[:, :, :2].reshape(-1, 2).mean(0)
        self.sd_ = X[:, :, :2].reshape(-1, 2).std(0) + 1e-9
        Z = self._prep(X)
        self.m = MLPRegressor(hidden_layer_sizes=(32, 8, 32), max_iter=200, random_state=self.seed).fit(Z, Z)

    def score_windows(self, X):
        Z = self._prep(X)
        return ((self.m.predict(Z) - Z) ** 2).mean(1)


class WaveShape(_RawWindowDetector):
    """attached report: amplitude-free waveform shape, shrinkage Mahalanobis"""
    agg = "top2"

    def __init__(self, N=10):
        self.W, self.stride = N, N // 2

    @staticmethod
    def _shape(X):
        Z = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-9)
        return Z.reshape(len(X), -1)

    def fit_windows(self, X):
        Z = self._shape(X)
        lw = LedoitWolf().fit(Z)
        self.loc_, self.P_ = lw.location_, np.linalg.inv(lw.covariance_)

    def score_windows(self, X):
        Z = self._shape(X) - self.loc_
        return np.einsum("ij,jk,ik->i", Z, self.P_, Z)


class SpecShape(_RawWindowDetector):
    """attached report: 0-5 Hz spectral SHAPE (demean, Hann, |rFFT|^2, drop DC, normalise), kNN"""
    agg = "top2"

    def __init__(self, N=10, k=5):
        self.W, self.stride, self.k = N, N // 2, k

    def _spec(self, X):
        Y = (X - X.mean(1, keepdims=True)) * np.hanning(self.W)[None, :, None]
        P = np.abs(np.fft.rfft(Y, axis=1)) ** 2
        P = P[:, 1:, :]
        P = P / (P.sum(1, keepdims=True) + 1e-12)
        return P.reshape(len(X), -1)

    def fit_windows(self, X):
        S = self._spec(X)
        if len(S) > 6000:
            S = S[np.random.default_rng(0).choice(len(S), 6000, replace=False)]
        self.nn = NearestNeighbors(n_neighbors=self.k).fit(S)

    def score_windows(self, X):
        return self.nn.kneighbors(self._spec(X))[0].mean(1)
