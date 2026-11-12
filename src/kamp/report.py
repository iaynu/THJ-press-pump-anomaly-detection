"""Aggregate results/raw/*.csv into result tables (results/tables) and figures (results/figures)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import spearmanr
from .features import FEATURES
from .models import FINAL, M2, M3

RAW, TAB, FIG = "results/raw", "results/tables", "results/figures"
KEY = [FINAL, M3, M2, "M1 통합 모델 (9특징 MCD)", "B2 kNN 거리", "B4 형상+스펙트럼 (첨부안, 오경보 1%로 보정)",
       "B4' 형상+스펙트럼 (첨부안 원래 규칙: 둘 다 q99)", "B3 진동 오토인코더"]


def _load(stem):
    """base run + follow-up runs (_m3: mixture model M3, _final: final model), when present"""
    proto, kind = stem.split("_", 1)
    parts = [pd.read_csv(f) for f in [f"{RAW}/{stem}.csv", f"{RAW}/{proto}_m3_{kind}.csv", f"{RAW}/{proto}_final_{kind}.csv"]
             if os.path.exists(f)]
    return pd.concat(parts, ignore_index=True)


def _font():
    for f in font_manager.findSystemFonts():
        if "NotoSansCJK" in f and "Regular" in f:
            font_manager.fontManager.addfont(f)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def _f1(tp, fp, fn):
    """precision is set to 0 when nothing is flagged (no true positives), so F1 = 0 in that case"""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def model_table(m):
    rows = []
    for (model), g in m.groupby("model", sort=False):
        nrm = g[g.set == "정상"]
        real = g[g.set == "실제 이상"]
        flt = g[g.family.notna()]
        pr = [_f1(r.tp, r.fp, r.fn) for r in real.itertuples()]
        pf = [_f1(r.tp, r.fp, r.fn) for r in flt.itertuples()]
        rows.append({
            "모델": model,
            "오경보율(정상 조각)": nrm.alarm.mean(), "오경보율 최대(폴드)": nrm.alarm.max(),
            "주의 단계 비율(정상)": nrm.attention.mean(),
            "실제 이상 재현율": real.recall.mean(), "실제 이상 정밀도": np.nanmean([p[0] for p in pr]),
            "실제 이상 F1": np.mean([p[2] for p in pr]), "실제 이상 ROC-AUC": real.auc.mean(), "실제 이상 PR-AUC": real.ap.mean(),
            "주입 고장 재현율(평균)": flt.recall.mean(), "주입 고장 F1(1:1, 평균)": np.mean([p[2] for p in pf]),
            "주입 고장 PR-AUC(평균)": flt.ap.mean(),
            "진동 고장 재현율": flt[flt.family == "진동"].recall.mean(), "전류 고장 재현율": flt[flt.family == "전류"].recall.mean(),
            "측정 튐 경보율": g[g.set == "측정 튐(1샘플)"].alarm.mean()})
    return pd.DataFrame(rows)


def main():
    _font()
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    M = {p: _load(f"{p}_metrics") for p in ["random", "blocks", "forward"] if os.path.exists(f"{RAW}/{p}_metrics.csv")}

    # ---- 1. model comparison (main protocol: random segment CV) + protocol robustness
    tabs = {p: model_table(m) for p, m in M.items()}
    t1 = tabs["random"]
    t1.round(3).to_csv(f"{TAB}/01_model_comparison_random.csv", index=False)
    prot = []
    for p, t in tabs.items():
        for _, r in t.iterrows():
            prot.append({"프로토콜": {"random": "무작위 조각 5-fold×2", "blocks": "시간 블록 5개", "forward": "순방향(앞80→뒤20)"}[p],
                         "모델": r["모델"], "오경보율": r["오경보율(정상 조각)"], "실제 이상 재현율": r["실제 이상 재현율"],
                         "주입 고장 재현율": r["주입 고장 재현율(평균)"], "주입 고장 F1": r["주입 고장 F1(1:1, 평균)"]})
    pd.DataFrame(prot).round(3).to_csv(f"{TAB}/02_protocol_robustness.csv", index=False)

    # per-fault recall
    m = M["random"]
    pf = m[m.family.notna()].groupby(["set", "model"], sort=False).recall.mean().unstack()
    pf = pf[[c for c in KEY if c in pf.columns]]
    pf.round(3).to_csv(f"{TAB}/03_per_fault_recall.csv")

    # confusion matrices of the final model (pooled over folds)
    fm = m[m.model == FINAL]
    real = fm[fm.set == "실제 이상"][["tp", "fn", "fp", "tn"]].sum()
    flt = fm[fm.family.notna()][["tp", "fn", "fp", "tn"]].sum()
    pd.DataFrame({"실제 이상 vs 정상": real, "주입 고장 vs 정상(1:1)": flt}).to_csv(f"{TAB}/04_confusion_final.csv")

    # robustness / falsification checks
    chk = m[(m.model == FINAL) & m.set.str.startswith("점검")].groupby("set")[["alarm", "recall"]].mean()
    base = m[(m.model == FINAL) & (m.set == "정상")].alarm.mean()
    chk.loc["기준: 변형 없음 (정상)"] = [base, np.nan]
    chk.round(3).to_csv(f"{TAB}/05_robustness_checks.csv")

    # ---- 2. error analysis of the final model
    S = _load("random_segments")
    S = S[S.model == FINAL]
    nrm = S[S.kind == "정상"].copy()
    nrm["alarm"] = nrm.level == 2
    nrm["조각 길이"] = pd.cut(nrm.n, [0, 7, 15, 23, 50], labels=["<8행", "8~15행", "16~23행", "24~50행"])
    nrm["직전 공백"] = pd.qcut(nrm.gap, 4, duplicates="drop")
    nrm["운전 상태"] = np.where(nrm.low_load, "저부하", "일반")
    fp_rows = []
    for col in ["운전 상태", "조각 길이", "직전 공백"]:
        for k, g in nrm.groupby(col, observed=True):
            fp_rows.append({"조건": col, "구간": str(k), "정상 조각 수": len(g), "오경보율": g.alarm.mean()})
    pd.DataFrame(fp_rows).round(4).to_csv(f"{TAB}/06_fp_by_condition.csv", index=False)

    flt_s = S[~S.kind.isin(["정상", "실제 이상"])].copy()
    flt_s["detected"] = flt_s.level == 2
    flt_s["운전 상태"] = np.where(flt_s.low_load, "저부하", "일반")
    fn = flt_s.groupby(["kind", "운전 상태"]).detected.mean().unstack()
    fn.round(3).to_csv(f"{TAB}/07_fault_recall_by_state.csv")

    det = flt_s[flt_s.detected]
    det["계열"] = np.where(det.kind.str.rsplit(" ", n=1).str[0].isin(
        ["진동 전체 증가", "충격 스파이크(16행당 1회)", "하부 진동 극성 반전"]), "진동 고장", "전류 고장")
    dg = pd.crosstab(det["계열"], det.diag, normalize="index")
    ro = S[S.kind == "실제 이상"]
    dg.loc["실제 이상"] = ro.diag.value_counts(normalize=True)
    dg.fillna(0).round(3).to_csv(f"{TAB}/08_diagnosis.csv")

    ccols = [f"c{j}" for j in range(9)]
    contrib = {}
    for name, g in [("실제 이상", ro), ("정상 중 오경보", nrm[nrm.alarm]), ("정상(경보 없음)", nrm[~nrm.alarm])]:
        c = g[ccols].clip(lower=0)
        contrib[name] = (c.div(c.sum(axis=1).replace(0, np.nan), axis=0)).mean()
    pd.DataFrame(contrib, index=ccols).set_axis(FEATURES).round(3).to_csv(f"{TAB}/09_feature_contribution_share.csv")

    aud = []
    for col, lab in [("gap", "직전 공백(초)"), ("n", "조각 길이(행)")]:
        rho, p = spearmanr(nrm.score, nrm[col], nan_policy="omit")
        aud.append({"메타데이터": lab, "Spearman rho": rho, "p값": p})
    t_min = (pd.to_datetime(nrm.t0) - pd.to_datetime(nrm.t0).min()).dt.total_seconds() / 60
    rho, p = spearmanr(nrm.score, t_min)
    aud.append({"메타데이터": "기록 경과 시간(분)", "Spearman rho": rho, "p값": p})
    pd.DataFrame(aud).round(4).to_csv(f"{TAB}/10_metadata_audit.csv", index=False)

    # ---- 3. streams (blocks)
    if os.path.exists(f"{RAW}/blocks_stream.csv"):
        st = _load("blocks_stream")
        cl = st[st.scenario == "정상 스트림"].groupby(["model", "rule"]).agg(
            경보_조각_비율=("alarm", "mean"), 시간당_경보_에피소드=("episodes_per_hour", "mean"),
            최대_연속_오경보=("max_run", "max"), 주의_단계_비율=("attention", "mean"))
        cl.round(3).to_csv(f"{TAB}/11_stream_false_alarms.csv")
        rp = st[st.scenario.isin(["진동 전체 증가", "충격 스파이크(16행당 1회)", "전류 오프셋 이동", "전류 진폭 증가",
                                  "전류 파형 왜곡(2.5Hz 성분)"])]
        rp.groupby(["scenario", "model", "rule"]).agg(첫_경보_크기_중앙값=("first_alarm_severity", "median"),
                                                     탐지율=("first_alarm_severity", lambda v: v.notna().mean()),
                                                     고장전_오경보=("false_before", "mean")).round(3).to_csv(f"{TAB}/12_early_detection.csv")
        ab = st[st.scenario == "실제 이상 갑자기 등장"].groupby(["model", "rule"]).delay_segments.agg(["median", "max"])
        ab.to_csv(f"{TAB}/13_abrupt_delay.csv")

    # ---- 4. label-noise (contamination) check and operating-state (regime) check
    TRIM = {"None": "보정 제외 없음", "3.0": "z>3 제외", "3.5": "z>3.5 제외 (최종)", "4.0": "z>4 제외",
            "5.0": "z>5 제외", "6.0": "z>6 제외"}
    if os.path.exists(f"{RAW}/trim_sweep_clean.csv"):
        d = pd.read_csv(f"{RAW}/trim_sweep_clean.csv", dtype={"trim": str}, keep_default_na=False)
        d["견고한 보정"] = d.trim.map(TRIM)
        d.groupby(["protocol", "견고한 보정"], sort=False).agg(
            제외된_학습조각_비율=("trimmed", "mean"), 오경보율=("false_alarm", "mean"),
            주입_고장_재현율=("fault_recall", "mean"), 실제_이상_재현율=("real_recall", "mean")) \
            .rename_axis(["프로토콜", "견고한 보정"]).round(4).to_csv(f"{TAB}/14a_trim_cost_on_clean_data.csv")
    parts = []
    for f, lab in [("trim_sweep_contam.csv", "M2 통합 + 전류 전용 (최종 구조)"), ("trim_sweep_contam_m3.csv", "M3 상태 혼합 통합 + 전류 전용")]:
        if os.path.exists(f"{RAW}/{f}"):
            d = pd.read_csv(f"{RAW}/{f}", dtype={"trim": str}, keep_default_na=False)
            d["모델"] = lab
            parts.append(d)
    if parts:
        d = pd.concat(parts)
        d["견고한 보정"] = d.trim.map(TRIM)
        d.groupby(["모델", "share", "견고한 보정"], sort=False).agg(
            제외된_학습조각_비율=("trimmed", "mean"), 오경보율=("false_alarm", "mean"),
            섞인_고장_재현율=("same_fault_recall", "mean"), 실제_이상_재현율=("real_recall", "mean")) \
            .rename_axis(["모델", "정상 라벨 오염 비율", "견고한 보정"]).round(3).to_csv(f"{TAB}/14b_label_noise_robustness.csv")
    if os.path.exists(f"{RAW}/contamination.csv"):
        d = pd.read_csv(f"{RAW}/contamination.csv")
        d.groupby(["model", "share"]).agg(오경보율=("false_alarm", "mean"), 섞인_고장_재현율=("same_fault_recall", "mean"),
                                        실제_이상_재현율=("real_recall", "mean")) \
            .rename_axis(["모델 (보정 제외 없음)", "정상 라벨 오염 비율"]).round(3).to_csv(f"{TAB}/14c_label_noise_M2_vs_M3.csv")
    if os.path.exists(f"{RAW}/regime_check.csv"):
        rg = pd.read_csv(f"{RAW}/regime_check.csv")
        rg["model"] = rg.model.str.split(" ").str[0].map({"M2": "M2 (MCD 통합)", "M3": "M3 (GMM3 통합)"})
        rg = rg.groupby("model").mean(numeric_only=True).drop(columns="fold")
        rg.rename(columns={"fa": "오경보율", "fa_lowload": "저부하 오경보율", "real": "실제 이상 재현율"}).T.round(3) \
            .to_csv(f"{TAB}/15_regime_check.csv")

    # ---- figures
    # F1 operating points
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for _, r in t1.iterrows():
        if r["모델"] == M2:      # identical to the final model on clean data (robust calibration trims nothing)
            continue
        final = r["모델"] == FINAL
        fx, fy = 100 * r["오경보율(정상 조각)"], 100 * r["주입 고장 재현율(평균)"]
        ax.scatter(fx, fy, s=120 if final else 60, color="#1f6feb" if final else "#888888", zorder=3)
        ax.annotate("최종 = M2 통합 + 전류 전용, 견고한 보정" if final else r["모델"].split(" (")[0].split(":")[0], (fx, fy),
                    textcoords="offset points", xytext=(-8, 8) if final else (6, 4), ha="right" if final else "left",
                    fontsize=9, color="#1f6feb" if final else "black")
    ax.axvline(1, color="#bbbbbb", ls="--", lw=1)
    ax.text(1.05, 2, "오경보 목표 1%", color="#777777", fontsize=8)
    ax.set_xlabel("정상 조각 오경보율 (%)")
    ax.set_ylabel("주입 고장 평균 재현율 (%)")
    ax.set_title("오경보율과 주입 고장 재현율 (무작위 조각 5-fold×2): 2관점 모델(최종, M3)이 가장 위")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG}/F1_operating_points.png", dpi=150)
    plt.close(fig)

    # F2 threshold sensitivity
    if os.path.exists(f"{RAW}/sensitivity.csv"):
        sv = pd.read_csv(f"{RAW}/sensitivity.csv").groupby("q").mean()
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(100 * sv.false_alarm, 100 * sv.fault_recall, "o-", color="#1f6feb", label="주입 고장 평균 재현율")
        ax.plot(100 * sv.false_alarm, 100 * sv.real_recall, "s--", color="#444444", label="실제 이상 재현율")
        for q, r in sv.iterrows():
            ax.annotate(f"q{q * 100:g}", (100 * r.false_alarm, 100 * r.fault_recall), textcoords="offset points",
                        xytext=(4, -12), fontsize=8, color="#555555")
        ax.set_xlabel("정상 조각 오경보율 (%)")
        ax.set_ylabel("재현율 (%)")
        ax.set_title("임계값을 낮추면 오경보가 늘면서 작은 고장도 잡힌다 (최종 모델)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{FIG}/F2_threshold_sensitivity.png", dpi=150)
        plt.close(fig)

    # F3 per-fault recall
    fig, ax = plt.subplots(figsize=(10, 6))
    sub = pf[[c for c in [FINAL, M3, "M1 통합 모델 (9특징 MCD)", "B4 형상+스펙트럼 (첨부안, 오경보 1%로 보정)"] if c in pf.columns]]
    y = np.arange(len(sub))
    h = 0.8 / sub.shape[1]
    colors = ["#1f6feb", "#7d8590", "#b1b8c0", "#e8a33d"]
    for j, c in enumerate(sub.columns):
        ax.barh(y + j * h, 100 * sub[c], height=h, color=colors[j], label=c)
    ax.set_yticks(y + 0.4 - h / 2)
    ax.set_yticklabels(sub.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("재현율 (%), 오경보 1% 기준")
    ax.set_title("고장 유형별 재현율")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG}/F3_per_fault_recall.png", dpi=150)
    plt.close(fig)

    # F4 two views
    fig, ax = plt.subplots(figsize=(7, 6))
    tf = lambda p: -np.log10(np.clip(1 - p, 1e-3, 1))
    groups = [("정상 시험 조각", S[S.kind == "정상"], "#b1b8c0", "o"),
              ("진동 전체 ×2", S[S.kind == "진동 전체 증가 2.0"], "#e8a33d", "^"),
              ("전류 중심선 +10", S[S.kind == "전류 오프셋 이동 10"], "#2da44e", "s"),
              ("실제 이상", S[S.kind == "실제 이상"], "#cf222e", "D")]
    for lab, g, c, mk in groups:
        ax.scatter(tf(g.p_joint), tf(g.p_current), s=14, alpha=0.6, color=c, marker=mk, label=lab)
    ax.axvline(2, color="#888888", ls="--", lw=1)
    ax.axhline(2, color="#888888", ls="--", lw=1)
    ax.set_xlabel("통합 관점 희귀도  −log10(1−p)")
    ax.set_ylabel("전류 관점 희귀도  −log10(1−p)")
    ax.set_title("진동 고장은 통합 관점, 전류 고장은 전류 관점이 잡는다")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG}/F4_two_views.png", dpi=150)
    plt.close(fig)

    # F5 confusion matrices
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (title, c) in zip(axs, [("실제 이상 vs 정상 (폴드 합)", real), ("주입 고장 vs 정상 1:1 (폴드 합)", flt)]):
        mat = np.array([[c.tn, c.fp], [c.fn, c.tp]])
        ax.imshow(mat, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(mat[i, j]):,}", ha="center", va="center", fontsize=12)
        ax.set_xticks([0, 1], ["예측 정상", "예측 이상(경고)"])
        ax.set_yticks([0, 1], ["실제 정상", "실제 이상"])
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG}/F5_confusion_final.png", dpi=150)
    plt.close(fig)

    # F6 robustness checks
    fig, ax = plt.subplots(figsize=(8, 3.8))
    vals = chk.alarm.fillna(chk.recall)
    ax.barh(vals.index, 100 * vals, color=["#cf222e" if "실제 이상" in i else "#7d8590" for i in vals.index])
    for i, v in enumerate(vals):
        ax.text(100 * v + 1, i, f"{100 * v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("경고 비율 (%)")
    ax.set_title("반증 점검: 변형을 가했을 때 최종 모델의 경고 비율")
    fig.tight_layout()
    fig.savefig(f"{FIG}/F6_robustness.png", dpi=150)
    plt.close(fig)

    # F7 stream timeline (one block)
    if os.path.exists(f"{RAW}/blocks_segments.csv"):
        b = _load("blocks_segments")
        b = b[(b.model == FINAL) & (b.kind == "정상") & (b.fold == 2)].copy()
        b["t"] = pd.to_datetime(b.t0)
        b = b.sort_values("t")
        fig, ax = plt.subplots(figsize=(10, 3.6))
        rar = tf(b.score)
        col = np.where(b.level == 2, "#cf222e", np.where(b.level == 1, "#e8a33d", "#7d8590"))
        ax.vlines(b.t, 0, rar, color=col, lw=1.5)
        ax.set_ylabel("희귀도 −log10(1−p)")
        ax.set_title("정상 스트림(시간 블록 3)의 조각별 점수: 회색 정상, 주황 주의, 빨강 경고. 빈 곳은 기록 공백")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(f"{FIG}/F7_stream_timeline.png", dpi=150)
        plt.close(fig)
    print("tables:", sorted(os.listdir(TAB)))
    print("figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
