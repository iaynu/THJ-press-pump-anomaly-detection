"""Synthetic faults injected into NORMAL test segments (rows x [AI0, AI1, AI2]) to measure how small a fault each model can detect. They are assumptions for stress-testing, not this machine's real fault shapes."""
import numpy as np

W_BLOCK = 16


def spikes(x, h, sd, rng):
    x = x.copy()
    for a in range(0, len(x), W_BLOCK):
        p = a + rng.integers(0, min(W_BLOCK, len(x) - a))
        x[p, 0] += h * sd[0] * rng.choice([-1, 1])
        x[p, 1] += h * sd[1] * rng.choice([-1, 1])
    return x


def lag(x, k):
    if k == 0:
        return x
    return np.column_stack([x[k:, 0], x[:-k, 1], x[k:, 2]])


def harmonic(x, r, rng):
    x = x.copy()
    amp = (x[:, 2].max() - x[:, 2].min()) / 2
    t = np.arange(len(x)) * 0.1
    x[:, 2] += r * amp * np.sin(2 * np.pi * 2.5 * t + rng.uniform(0, 2 * np.pi))
    return x


def cur_amp(x, k):
    x = x.copy()
    mid = (x[:, 2].max() + x[:, 2].min()) / 2
    x[:, 2] = mid + (x[:, 2] - mid) * k
    return x


def scale(x, k, ch):
    x = x.copy()
    x[:, ch] *= k
    return x


def shift(x, d):
    x = x.copy()
    x[:, 2] += d
    return x


FAULTS = {
    # name: (unit label, severities, fn(x, sev, sd, rng))
    "진동 전체 증가": ("배", [1.1, 1.25, 1.5, 2.0, 3.0], lambda x, v, sd, r: scale(x, v, [0, 1])),
    "상부 진동만 증가": ("배", [1.1, 1.25, 1.5, 2.0, 3.0], lambda x, v, sd, r: scale(x, v, [0])),
    "충격 스파이크(16행당 1회)": ("×정상 표준편차", [1, 2, 3, 5, 8], lambda x, v, sd, r: spikes(x, v, sd, r)),
    "하부 진동 위상 지연": ("행(0.1초)", [1, 2, 3, 4], lambda x, v, sd, r: lag(x, int(v))),
    "하부 진동 극성 반전": ("", [1], lambda x, v, sd, r: scale(x, -1, [1])),
    "전류 오프셋 이동": ("단위", [2, 5, 10, 20, 40], lambda x, v, sd, r: shift(x, v)),
    "전류 진폭 증가": ("배", [1.05, 1.1, 1.2, 1.3, 1.5], lambda x, v, sd, r: cur_amp(x, v)),
    "전류 파형 왜곡(2.5Hz 성분)": ("×진폭", [0.02, 0.05, 0.1, 0.2, 0.3], lambda x, v, sd, r: harmonic(x, v, r)),
}

