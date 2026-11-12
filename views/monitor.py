"""Monitoring replay page: one segment at a time through the dashboard model (fitted on the first 80% of normal data only),
2-stage alarm, trend alarm and a simulated stop request after K consecutive alarms."""
import os

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

from kamp import dashboard_core as dc
from kamp import ui
from kamp.data import load_segments
from kamp.features import FEATURES

NAME = ui.LEVEL_NAME
ss = st.session_state


@st.cache_resource(show_spinner="대시보드 모델과 기록을 불러오는 중...")
def resources():
    if not os.path.exists(dc.PATH):
        dc.build()
    B = dc.load()
    return B, load_segments("data/press_data_normal.csv", 0), load_segments("data/outlier_data.csv", 1)


B, N, O = resources()
S = B["system"]
TH = {"attention": float(S.t_att), "warning": float(S.t_warn), "rarity_floor": 1 / (max(len(r) for r in S.ref.values()) + 1)}

for k, v in dict(scn=list(dc.SCENARIOS)[0], stop_k=dc.STOP_K, steps=[], results=[], pos=0, running=False, mon=None, log=[],
                 speed=1.0, auto=True, truth=True, starts=None).items():
    ss.setdefault(k, v)


def reset():
    ss.steps = dc.make_stream(ss.scn, N, O, B)
    ss.starts = ui.stream_positions(ss.steps)
    ss.results, ss.pos, ss.log, ss.running = [], 0, [], False
    ss.mon = dc.Monitor(B, ss.stop_k, ss.auto)


def on_scenario():
    ss.scn = ss.w_scn
    reset()


def on_k():
    ss.stop_k = ss.w_k
    reset()


def ack():
    ss.mon.acknowledge()
    ss.running = ss.pos < len(ss.steps)


def advance():
    if ss.pos >= len(ss.steps) or ss.mon.stopped:
        return
    step = ss.steps[ss.pos]
    was_stopped = ss.mon.stopped
    r = ss.mon.step(step.x)
    ss.results.append(r)
    ss.pos += 1
    if (r.level >= 1 and r.judged) or r.trend or (r.stop and not was_stopped):
        top = "" if r.contrib is None else FEATURES[int(np.argmax(r.contrib))]
        action = "정지 요청" if r.stop and not was_stopped else (NAME[r.level] if r.level >= 1 and r.judged else "추세 경보")
        ss.log.insert(0, {"조각": r.i + 1, "행 수": len(step.x), "단계": NAME[r.level] if r.judged else "보류",
                          "조치": action + (f" ({r.reason})" if r.reason else ""),
                          "희귀도": "-" if np.isnan(r.pct) else ui.fmt_rarity(1 - r.pct, TH["rarity_floor"]),
                          "신호 유형": r.diag, "주요 특징": top, "시나리오 정답": step.truth})


if ss.mon is None:
    reset()
onset = next((i for i, s in enumerate(ss.steps) if s.truth.startswith("이상")), None)

# ------------------------------------------------------------------------------------------------ header + controls
st.markdown(ui.head("모니터링 시연", "기록을 조각 단위로 재생하며 경보와 정지 요청이 어떻게 나는지 보여 줍니다 · 설비와 연결되어 있지 않습니다"),
            unsafe_allow_html=True)
st.markdown(f'<p class="note" style="margin:2px 0 10px 0">이 화면은 시연용 모델을 씁니다. 시연 모델은 정상 {len(N)}조각 중 시간상 앞 '
            f'{B["n_train"]}조각(80%)으로만 학습하고, 학습에 쓰지 않은 뒤 {len(B["holdout"])}조각과 실제 이상 기록을 재생합니다. '
            f'제출 모델(run.py train으로 학습, 파일 분석 화면에서 사용)은 정상 데이터 {len(N)}조각 전체로 학습합니다.</p>',
            unsafe_allow_html=True)
with st.container(border=True):
    c = st.columns([2.4, 1.3, 1.3, 1, 1])
    c[0].selectbox("시나리오", list(dc.SCENARIOS), index=list(dc.SCENARIOS).index(ss.scn), key="w_scn", on_change=on_scenario)
    c[1].select_slider("정지 조건 K (연속 조각)", [1, 2, 3], value=ss.stop_k, key="w_k", on_change=on_k,
                       help="경고 또는 추세 경보가 K조각 연속되면 정지를 요청합니다. 1은 즉시, 클수록 오정지가 줄지만 늦어집니다.")
    ss.speed = c[2].select_slider("재생 간격", [0.5, 1.0, 1.5, 2.0, 3.0], value=ss.speed, format_func=lambda v: f"{v}초/조각")
    ss.auto = c[3].toggle("자동 정지", value=ss.auto)
    ss.truth = c[4].toggle("정답 표시", value=ss.truth, help="시나리오에서 이상이 시작되는 조각을 표시합니다. 실제 설비에는 없는 정보입니다.")
    ss.mon.auto_stop = ss.auto
    comp = (f"구성: 1~{onset}번째 조각은 학습에 쓰지 않은 정상 기록, {onset + 1}~{len(ss.steps)}번째 조각은 "
            + ("실제 이상 기록(outlier_data.csv 21조각)" if ss.scn == "실제 이상 기록" else "같은 정상 기록에 고장을 주입한 조각")
            if onset is not None else f"구성: 학습에 쓰지 않은 정상 기록 {len(ss.steps)}조각")
    lead = "" if ss.scn == "실제 이상 기록" else dc.SCENARIOS[ss.scn] + " "
    st.markdown(f'<p class="note" style="margin:0">{ui.esc(lead + comp)}.</p>', unsafe_allow_html=True)
    b = st.columns([1, 1, 1, 1, 3])
    done = ss.pos >= len(ss.steps)
    b[0].button("일시정지" if ss.running else ("다시 재생" if done else "재생"), type="primary", width="stretch",
                on_click=lambda: (reset() if done else None, ss.update(running=not ss.running)))
    b[1].button("한 조각", width="stretch", on_click=advance, disabled=done or ss.mon.stopped or ss.running)
    b[2].button("처음부터", width="stretch", on_click=reset)
    b[3].button("정지 해제", width="stretch", on_click=ack, disabled=not ss.mon.stopped, help="점검을 마쳤다고 보고 정지 요청을 해제합니다.")


# ------------------------------------------------------------------------------------------------ live view
def view():
    n = ss.pos
    if ss.running and not ss.mon.stopped and n < len(ss.steps):
        advance()
        n = ss.pos
        if ss.mon.stopped or n >= len(ss.steps):
            ss.running = False
            st.rerun()
    last = ss.results[-1] if ss.results else None
    stopped = ss.mon.stopped
    live = ss.running and not stopped

    if last is None:
        state, tone, desc = "대기", None, "재생을 누르면 첫 조각부터 판정합니다."
    elif stopped:
        k = len(ss.results) - 1
        while k > 0 and ss.results[k - 1].stop:
            k -= 1
        r0 = ss.results[k]
        state, tone = "정지 요청", 2
        desc = (f"{r0.i + 1}번째 조각에서 {r0.reason or '연속 조건 충족'} · 신호 유형 {r0.diag or last.diag or '-'}. "
                "점검 후 '정지 해제'를 누르면 재개됩니다.")
    elif not last.judged:
        state, tone, desc = "판정 보류", None, f"{len(ss.steps[n - 1].x)}행 조각이라 판정하지 않고 이전 상태({NAME[last.level]})를 유지합니다."
    else:
        state, tone = NAME[last.level], last.level
        desc = f"경고 연속 {last.consec_warn}/{ss.stop_k}" + (f" · 추세 경보 연속 {last.consec_trend}/{ss.stop_k}" if last.trend else "")
    rar = "-" if not last or not last.judged else ui.fmt_rarity(1 - last.pct, TH["rarity_floor"])
    st.markdown(ui.band(state, ui.esc(desc), tone, [
        ("프레스(시뮬레이션)", "정지 요청" if stopped else "가동", "자동 정지 " + ("사용" if ss.auto else "끔")),
        ("현재 조각 희귀도", rar, "정상 대비 상위"),
        ("진행", f"{n} / {len(ss.steps)}", "재생 중" if live else ("완료" if n >= len(ss.steps) else ("일시정지" if n else "대기"))),
        ("경보 기록", f"{len(ss.log)}", "주의·경고·추세·정지")]), unsafe_allow_html=True)
    st.progress(n / max(len(ss.steps), 1))

    shown_onset = onset if ss.truth else None
    stops = [r.i for r in ss.results if r.stop and (r.i == 0 or not ss.results[r.i - 1].stop)]
    cfg = {"staticPlot": True} if live else {"displaylogo": False, "displayModeBar": False}
    left, right = st.columns([3, 2], gap="medium")
    with left:
        st.markdown("### 실시간 신호")
        st.markdown(ui.legend([("정상", ui.LEVEL_COL[0]), ("주의", ui.LEVEL_COL[1]), ("경고", ui.LEVEL_COL[2]),
                               ("판정 보류(8행 미만)", ui.HOLD)]) + '<p class="note" style="margin:0">음영으로 조각을 구분하고 위에 조각 번호를 표시합니다. 가로 폭이 곧 실제 측정 길이입니다.</p>',
                    unsafe_allow_html=True)
        if n:
            st.plotly_chart(ui.monitor_signal(ss.steps, ss.results, ss.starts, shown_onset, hover=not live),
                            width="stretch", key="sig", config=cfg)
        else:
            st.markdown('<div class="panel" style="height:430px;display:flex;align-items:center;justify-content:center">'
                        '<span class="note">재생을 시작하면 최근 40초 분량의 신호가 표시됩니다.</span></div>', unsafe_allow_html=True)
    with right:
        st.markdown("### 조각별 희귀도")
        st.markdown(ui.legend([("조각 판정", ui.LEVEL_COL[0]), ("추세(EWMA) 점선", ui.MUTED)]), unsafe_allow_html=True)
        st.plotly_chart(ui.monitor_score(ss.steps, ss.results, TH, B["trend"], shown_onset, stops), width="stretch",
                        key="score", config=cfg)

    a, b2 = st.columns([2, 3], gap="medium")
    with a:
        st.markdown("### 원인 진단")
        alerted = [r for r in ss.results if r.judged and r.contrib is not None]
        cur = last if last is not None and last.contrib is not None else (alerted[-1] if alerted else None)
        if cur is None:
            st.markdown('<div class="panel"><span class="note">아직 주의·경고 조각이 없습니다. 경보가 나면 어떤 특징이 점수를 올렸는지 여기에 표시됩니다.</span></div>',
                        unsafe_allow_html=True)
        else:
            c = np.clip(cur.contrib, 0, None)
            c = c / c.sum() if c.sum() > 0 else c
            st.markdown(f'<div class="kv"><span class="k">조각</span><span>{cur.i + 1}' + ("" if cur is last else " (마지막 경보)")
                        + f'</span><span class="k">신호 유형</span><span>{ui.esc(cur.diag)}</span>'
                        f'<span class="k">관점별 희귀도</span><span>통합 {ui.fmt_rarity(1 - cur.p_joint, TH["rarity_floor"])} · '
                        f'전류 전용 {ui.fmt_rarity(1 - cur.p_current, TH["rarity_floor"])}</span></div>', unsafe_allow_html=True)
            st.plotly_chart(ui.contrib_fig(c, 250), width="stretch", key="contrib", config=cfg)
    with b2:
        st.markdown("### 경보 기록")
        cols = ["조각", "행 수", "단계", "조치", "희귀도", "신호 유형", "주요 특징", "시나리오 정답"]
        d = pd.DataFrame(ss.log, columns=cols)
        if not ss.truth:
            d = d.drop(columns="시나리오 정답")
        st.dataframe(d, hide_index=True, width="stretch", height=330)


if ss.running and not ss.mon.stopped and ss.pos < len(ss.steps):
    st.fragment(view, run_every=ss.speed)()
else:
    st.fragment(view)()

# ------------------------------------------------------------------------------------------------ reference
with st.expander("성능 요약과 한계"):
    p1 = "results/tables/01_model_comparison_random.csv"
    if os.path.exists(p1):
        t1 = pd.read_csv(p1)
        f = t1[t1["모델"].str.startswith("최종")].iloc[0]
        st.markdown(ui.band("최종 모델 교차검증", "무작위 5-fold×2, 학습에 쓰지 않은 조각만 평가", None, [
            ("정상 오경보율", f"{float(f['오경보율(정상 조각)']):.1%}", "정상 조각"),
            ("실제 이상 재현율", f"{float(f['실제 이상 재현율']):.0%}", "21조각"),
            ("주입 고장 재현율", f"{float(f['주입 고장 재현율(평균)']):.1%}", "11종 평균"),
            ("전류 1샘플 튐 경보", f"{float(f['측정 튐 경보율']):.1%}", "낮을수록 좋음")]), unsafe_allow_html=True)
    p2 = "results/raw/dashboard_stop_rule.csv"
    if os.path.exists(p2):
        sr = pd.read_csv(p2)
        st.markdown("### 정지 규칙 점검")
        st.caption("경고 연속 조각 수(K)별 오정지 비율과 고장 시작 후 정지까지의 조각 수(중앙값)")
        g = sr.groupby(["scenario", "stop_k"]).agg(오정지=("false_stop", "mean"), 지연=("stop_delay", "median"),
                                                   정지못함=("first_stop", lambda s: s.isna().mean())).reset_index()
        g["오정지"] = g["오정지"].map("{:.0%}".format)
        g["정지못함"] = g["정지못함"].map("{:.0%}".format)
        g["지연"] = g["지연"].map(lambda v: "-" if pd.isna(v) else f"{v:.0f}")
        st.dataframe(g.rename(columns={"scenario": "시나리오", "stop_k": "K"}), hide_index=True, width="stretch")
        st.caption("정상·노이즈 시나리오에서 '정지못함 100%'는 올바른 결과입니다. 극성 반전은 미탐지 한계가 그대로 나타납니다.")
    st.markdown('<p class="note">시연 모델은 정상 조각의 시간상 앞 80%로만 학습했고, 재생은 학습에 쓰지 않은 뒤 20%와 실제 이상 21조각, '
                '주입 고장을 사용합니다. 조각이 끝나야 판정하므로 조각 길이(최대 5초)만큼 지연이 있습니다. 실제 이상은 하루치 21조각뿐이고 '
                '주입 고장은 가정이며, 극성 반전·저부하 진동 증가·센서 이득 변화는 약점입니다.</p>', unsafe_allow_html=True)
