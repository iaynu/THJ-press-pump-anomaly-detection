"""Model catalogue: every candidate compared in the evaluation, as calibrated Systems."""
from functools import partial
from .detectors import FeatureDetector, MCD, GMM, IForest, KNN, RMSRule, VibAE, WaveShape, SpecShape
from .features import VIB, CUR
from .system import System

JOINT = partial(FeatureDetector, MCD, None, name="joint")
CURRENT = partial(FeatureDetector, MCD, CUR, name="current")
VIBONLY = partial(FeatureDetector, MCD, VIB, name="vibration")
JOINT_GMM = partial(FeatureDetector, partial(GMM, 3), None, name="joint_gmm")


def catalogue(seed=0):
    return {
        "B0 진동 RMS 규칙": lambda: System({"rms": partial(FeatureDetector, RMSRule, None)}),
        "B1 Isolation Forest": lambda: System({"if": partial(FeatureDetector, IForest, None)}),
        "B2 kNN 거리": lambda: System({"knn": partial(FeatureDetector, KNN, None)}),
        "B3 진동 오토인코더": lambda: System({"ae": partial(VibAE, seed)}),
        "B4 형상+스펙트럼 (첨부안, 오경보 1%로 보정)": lambda: System({"wave": WaveShape, "spec": SpecShape}, fusion="max"),
        "B4' 형상+스펙트럼 (첨부안 원래 규칙: 둘 다 q99)": lambda: System({"wave": WaveShape, "spec": SpecShape}, rule="and_each"),
        "M1 통합 모델 (9특징 MCD)": lambda: System({"joint": JOINT}),
        "M2 통합 + 전류 전용 (2관점)": lambda: System({"joint": JOINT, "current": CURRENT}, fusion="max"),
        "M3 상태 혼합 통합 + 전류 전용 (2관점)": lambda: System({"joint": JOINT_GMM, "current": CURRENT}, fusion="max"),
        "최종: M2 + 견고한 임계값 보정": final_system,
    }


M2 = "M2 통합 + 전류 전용 (2관점)"
M3 = "M3 상태 혼합 통합 + 전류 전용 (2관점)"
FINAL = "최종: M2 + 견고한 임계값 보정"
DETAIL = (M2, M3, FINAL)


# final system: same two views, with consensus glitch repair of the current signal (see glitch_fix.py)
JOINT_FINAL = partial(FeatureDetector, MCD, None, name="joint", repair="seg_both", tau=0.3)
CURRENT_FINAL = partial(FeatureDetector, MCD, CUR, name="current", repair="seg_both", tau=0.3)


def final_system():
    """two robust views (joint 9-feature MCD + current-only MCD), max fusion, thresholds from out-of-fold
    normal scores after trimming suspected label noise (robust log-score z > 3.5; trims nothing on the clean data,
    protects the thresholds when up to ~5% of the normal label is actually faulty); a wild single current sample is repaired
    by a segment-wide leave-one-out vote before windows are cut"""
    return System({"joint": JOINT_FINAL, "current": CURRENT_FINAL}, fusion="max", trim_z=3.5)
