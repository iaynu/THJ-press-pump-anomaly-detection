"""Side-by-side tables: the Colab notebook's final model (A_physics / none, as run) vs our models, same split and tests.
Reads results/notebook_compare/ours_*.csv (kamp.experiments.notebook_compare) and the notebook's own outputs + theirs_*.csv
(see README.txt in the folder)."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUT = "results/notebook_compare"
NB = T = f"{OUT}/notebook_outputs"
LAB = "노트북 최종 (물리특징 MCD 단독)"


def auc(c, f):
    c, f = np.asarray(c, float), np.asarray(f, float)
    return roc_auc_score(np.r_[np.zeros(len(c)), np.ones(len(f))], np.r_[c, f])


def main():
    ours = pd.read_csv(f"{OUT}/ours_notebook_protocol.csv")
    seg = pd.read_csv(f"{NB}/theirs_segments.csv")
    rec = pd.read_csv(f"{T}/recorded_evaluation.csv").query("level=='segment'").iloc[0]
    temp = pd.read_csv(f"{T}/temporal_normal_evaluation.csv").query("candidate=='A_physics' and policy=='none'").iloc[0]
    nrm = seg[(seg.source == "normal") & (seg.split == "normal_test") & seg.direct_evidence]
    ab = seg[(seg.source == "abnormal") & seg.direct_evidence]
    th = dict(model=LAB, false_warning_segment_rate=temp.normal_false_warning_segment_rate,
              attention_segment_rate=temp.normal_attention_segment_rate, n_direct_normal=len(nrm),
              abnormal_direct_recall=rec.recall_direct, abnormal_all_recall=(seg[seg.source == "abnormal"].warning.mean()),
              precision_mixture=rec.precision_mixture, f1_mixture=rec.f1_mixture,
              tp=rec.true_positive, fn=rec.false_negative, fp=rec.false_positive, tn=rec.true_negative,
              real_vs_normal_auc=auc(nrm.candidate_score, ab.candidate_score))
    t1 = pd.concat([pd.DataFrame([th]), ours], ignore_index=True)
    t1.round(3).to_csv(f"{OUT}/01_recorded_data_comparison.csv", index=False)

    # their five perturbations and falsification probes
    syn = pd.read_csv(f"{T}/synthetic_sensitivity.csv").set_index("test").warning_segment_rate
    fal = pd.read_csv(f"{T}/falsification_results.csv").set_index("test").warning_segment_rate
    for name, src, f in [("02_notebook_perturbations.csv", "perturb", syn), ("03_falsification_probes.csv", "falsify", fal)]:
        o = pd.read_csv(f"{OUT}/ours_{src}.csv").pivot(index="test", columns="model", values="warning_rate")
        o.insert(0, LAB, f.reindex(o.index))
        o.loc["평균"] = o.mean()
        o.round(3).to_csv(f"{OUT}/{name}")

    # our 11 injected faults through both
    tf = pd.read_csv(f"{NB}/theirs_on_our_faults.csv")
    base = tf[tf.kind == "정상"].set_index("segment_id")
    rows = []
    for kind, g in tf[tf.kind != "정상"].groupby("kind", sort=False):
        g = g.set_index("segment_id")
        rows.append(dict(test=kind, **{LAB + " 경고율": g.warning.mean(), LAB + " AUC": auc(base.loc[g.index, "margin"], g.margin)}))
    th2 = pd.DataFrame(rows).set_index("test")
    of = pd.read_csv(f"{OUT}/ours_faults.csv")
    w = of.pivot(index="test", columns="model", values="warning_rate").add_suffix(" 경고율")
    a = of.pivot(index="test", columns="model", values="auc_vs_clean").add_suffix(" AUC")
    t4 = th2.join(w).join(a)
    t4 = t4.reindex([k for k in th2.index])
    t4.loc["평균"] = t4.mean()
    t4.round(3).to_csv(f"{OUT}/04_injected_faults.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .report import _font
    _font()
    fin = [c for c in w.columns if c.startswith("최종")][0]
    d = pd.DataFrame({"노트북 최종": th2[LAB + " 경고율"], "우리 최종": w[fin]}).drop("평균", errors="ignore")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(y - 0.2, 100 * d["노트북 최종"], 0.4, color="#e8a33d", label="노트북 최종")
    ax.barh(y + 0.2, 100 * d["우리 최종"], 0.4, color="#1f6feb", label="우리 최종")
    ax.set_yticks(y, d.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("경고 비율 (%) — 각 모델의 자기 경고 기준, 같은 시험 조각 96개")
    ax.set_title("같은 분할·같은 주입 고장: 노트북은 아무것도 경고하지 못한다")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/F8_notebook_vs_ours_injected_faults.png", dpi=150)
    plt.close(fig)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(t1.round(3).T.to_string())
    for f in ["02_notebook_perturbations", "03_falsification_probes"]:
        print("\n", f); print(pd.read_csv(f"{OUT}/{f}.csv", index_col=0).round(3).to_string())
    print("\n04"); print(t4.round(3).to_string())


if __name__ == "__main__":
    main()
