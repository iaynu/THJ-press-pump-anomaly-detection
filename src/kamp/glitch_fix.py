"""Can the 1-sample current glitch warnings be reduced without losing detection?
Variants of the final structure that differ only in the glitch repair of the current signal.
Random segment 5-fold, seed 0. Writes results/raw/glitch_fix.csv
  glitch        one wild current sample of +0.5 amplitude   (should NOT warn)
  glitch_small  +0.25 amplitude
  glitch_two    two such samples"""
from .port import main

VARIANTS = {
    "G0 기준 (4σ 규칙)": {},
    "G1 하나 빼기 복원 (τ=0.5)": dict(repair="loo", tau=0.5),
    "G2 하나 빼기 복원 (τ=0.3)": dict(repair="loo", tau=0.3),
    "G3 하나 빼기 복원 (τ=0.15)": dict(repair="loo", tau=0.15),
    "G4 하나 빼기 + 4σ 규칙 (τ=0.3)": dict(repair="both", tau=0.3),
    "G5 조각 전체 합의 복원 + 4σ 규칙": dict(repair="seg_thr", tau=0.3),
    "G6 조각 전체 합의 복원 + 하나 빼기 + 4σ 규칙": dict(repair="seg_both", tau=0.3),
}
RUN = ["G5 조각 전체 합의 복원 + 4σ 규칙", "G6 조각 전체 합의 복원 + 하나 빼기 + 4σ 규칙"]

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "new":      # only the variants added after the first run
        main(variants={k: VARIANTS[k] for k in RUN}, out="results/raw/glitch_fix2.csv")
    else:
        main(variants={k: v for k, v in VARIANTS.items() if k not in RUN}, out="results/raw/glitch_fix.csv")
