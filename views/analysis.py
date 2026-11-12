"""File analysis page: choose data -> check -> analyse -> results."""
import hashlib
import io
import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

from kamp import ui
from kamp.analysis import analyze, read_csv, validate, InputError, MIN_ROWS
from kamp.data import segment_table

MODEL = "models/final_system.pkl"
SAMPLES = {"이상 기록 (outlier_data.csv · 600행)": "data/outlier_data.csv",
           "정상 기록 (press_data_normal.csv · 20,000행)": "data/press_data_normal.csv"}
STEPS = ["데이터 선택", "데이터 확인", "분석", "결과"]
ss = st.session_state
ss.setdefault("fa_stage", "select")       # select -> review -> done
for k in ("fa_name", "fa_bytes", "fa_df", "fa_rep", "fa_err", "fa_res", "fa_hash", "fa_preview"):
    ss.setdefault(k, None)


@st.cache_resource(show_spinner="모델을 불러오는 중...")
def model():
    if not os.path.exists(MODEL):
        raise FileNotFoundError("models/final_system.pkl이 없습니다. 먼저 `python run.py train`을 실행하세요.")
    with open(MODEL, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False, max_entries=4)
def run_analysis(data: bytes, digest: str):
    return analyze(read_csv(io.BytesIO(data)), model())


@st.cache_data(show_spinner="그래프를 그리는 중...", max_entries=12)
def timeline(digest: str, xmode: str, focus, _R):
    return ui.file_timeline(_R.rows, _R.segments, _R.thresholds, xmode, focus)


def reset():
    for k in ("fa_name", "fa_bytes", "fa_df", "fa_rep", "fa_err", "fa_res", "fa_hash", "fa_preview"):
        ss[k] = None
    ss.fa_stage = "select"
    ss.pop("fa_upload", None)
    ss.pop("fa_focus", None)


def load(name, data):
    """read + validate only (fast); errors are kept for the error state"""
    ss.fa_name, ss.fa_bytes, ss.fa_res, ss.fa_err = name, data, None, None
    ss.fa_hash = hashlib.sha1(data).hexdigest()
    try:
        d = read_csv(io.BytesIO(data))
        clean, rep = validate(d)
        segs = segment_table(clean.assign(Equipment_state=-1))
        n = np.array([s.n for s in segs])
        rep.update(segments=len(segs), segments_short=int((n < MIN_ROWS).sum()),
                   t0=clean.TimeStamp.min(), t1=clean.TimeStamp.max())
        ss.fa_df, ss.fa_rep, ss.fa_stage = d, rep, "review"
    except InputError as e:
        ss.fa_err, ss.fa_stage = str(e), "select"


def on_upload():
    f = ss.get("fa_upload")
    if f is not None:
        load(f.name, f.getvalue())


def use_sample(label):
    with open(SAMPLES[label], "rb") as f:
        load(os.path.basename(SAMPLES[label]), f.read())


# ------------------------------------------------------------------------------------------------ header
st.markdown(ui.head("프레스 유압펌프 이상탐지", "진동 2채널·전류 기록을 연속 구간(조각) 단위로 판정합니다"), unsafe_allow_html=True)
cur = {"select": 0, "review": 1, "done": 3}[ss.fa_stage]
st.markdown(ui.stepper(STEPS, cur, done_upto=cur - 1 if ss.fa_stage != "done" else 3), unsafe_allow_html=True)

try:
    B = model()
except FileNotFoundError as e:
    st.markdown(f'<div class="panel err"><div class="t">모델 파일이 없습니다</div><div class="note">{ui.esc(e)}</div></div>',
                unsafe_allow_html=True)
    st.stop()

# ------------------------------------------------------------------------------------------------ 1. select (empty / error)
if ss.fa_stage == "select":
    if ss.fa_err:
        st.markdown(f'<div class="panel err"><div class="t">이 파일은 분석할 수 없습니다 · {ui.esc(ss.fa_name)}</div>'
                    f'<div class="note">{ui.esc(ss.fa_err)}</div>'
                    '<div class="note" style="margin-top:6px">열 이름과 형식을 확인한 뒤 다른 파일을 선택하세요.</div></div>',
                    unsafe_allow_html=True)
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("### 분석할 기록 선택")
        st.file_uploader("CSV 파일", type=["csv"], key="fa_upload", on_change=on_upload,
                         help="대회 제공 파일과 같은 형식의 CSV")
        st.caption("또는 대회 제공 기록으로 바로 시작합니다.")
        c1, c2 = st.columns(2)
        labels = list(SAMPLES)
        c1.button(labels[0], width="stretch", on_click=use_sample, args=(labels[0],))
        c2.button(labels[1], width="stretch", on_click=use_sample, args=(labels[1],))
    with right:
        st.markdown("### 입력 형식")
        st.markdown(
            '<div class="kv">'
            '<span class="k">TimeStamp</span><span>측정 시각 (약 0.1초 간격)</span>'
            '<span class="k">AI0_Vibration</span><span>상부 진동</span>'
            '<span class="k">AI1_Vibration</span><span>하부 진동</span>'
            '<span class="k">AI2_Current</span><span>전류</span>'
            '<span class="k">Equipment_state</span><span>선택. 모델 입력으로 쓰지 않음</span></div>'
            f'<p class="note" style="margin-top:10px">시각 간격이 0.5초를 넘으면 다른 조각으로 나눕니다. 판정은 조각 단위이며 '
            f'{MIN_ROWS}행 미만 조각은 시간상 가장 가까운 조각의 판정을 따릅니다. 첫 열이 index이면 원본 행 번호로 사용합니다.</p>',
            unsafe_allow_html=True)
    st.stop()

rep = ss.fa_rep

# ------------------------------------------------------------------------------------------------ 2. review + run
if ss.fa_stage == "review":
    st.markdown("### 데이터 확인")
    span = rep["t1"] - rep["t0"]
    st.markdown(ui.band(
        ss.fa_name, f"{rep['t0']:%Y-%m-%d %H:%M:%S} ~ {rep['t1']:%H:%M:%S} (기록 범위 {span.total_seconds() / 60:.1f}분)", None,
        [("전체 행", f"{rep['rows_total']:,}", "파일의 데이터 행"),
         ("분석 대상 행", f"{rep['rows_clean']:,}", f"중복 {rep['rows_duplicate']} · 제외 {rep['rows_invalid']}"),
         ("연속 구간(조각)", f"{rep['segments']:,}", f"{MIN_ROWS}행 미만 {rep['segments_short']}개")]), unsafe_allow_html=True)
    notes = []
    if rep["rows_duplicate"]:
        notes.append(f"시각과 센서 값이 모두 같은 중복 행 {rep['rows_duplicate']}개는 한 번만 분석하고, 결과 표에서는 원래 행의 판정을 그대로 붙입니다"
                     f" (예: index {', '.join(map(str, rep['duplicate_index'][:5]))}).")
    if rep["rows_invalid"]:
        notes.append(f"값이 비었거나 숫자가 아닌 행 {rep['rows_invalid']}개는 분석에서 제외하고 결과 표에 '제외'로 남깁니다"
                     f" (예: index {', '.join(map(str, rep['invalid_index'][:5]))}). 제외된 행 앞뒤는 다른 조각으로 나눕니다.")
    if notes:
        st.markdown('<p class="note">' + "<br>".join(ui.esc(n) for n in notes) + "</p>", unsafe_allow_html=True)
    st.dataframe(ss.fa_df.head(8), width="stretch", height=250)
    c1, c2, _ = st.columns([1.2, 1.2, 4])
    go_run = c1.button("분석 실행", type="primary", width="stretch")
    c2.button("다른 파일 선택", width="stretch", on_click=reset)
    if go_run:
        with st.status("분석 중", expanded=True) as status:
            st.write(f"조각 {rep['segments']:,}개로 나누고 윈도우별 특징 9개를 계산합니다.")
            st.write("통합·전류 전용 두 관점의 정상 대비 거리를 계산하고 주의·경고를 판정합니다.")
            try:
                ss.fa_res = run_analysis(ss.fa_bytes, ss.fa_hash)
            except InputError as e:
                status.update(label="분석할 수 없습니다", state="error")
                ss.fa_err, ss.fa_stage = str(e), "select"
                st.rerun()
            except Exception as e:                                   # noqa: BLE001
                status.update(label="분석 중 오류가 발생했습니다", state="error")
                st.markdown(f'<div class="panel err"><div class="t">분석 오류</div><div class="note">{ui.esc(type(e).__name__)}: '
                            f'{ui.esc(e)}</div></div>', unsafe_allow_html=True)
                st.stop()
            status.update(label="분석 완료", state="complete")
        ss.fa_stage = "done"
        st.rerun()
    st.stop()

# ------------------------------------------------------------------------------------------------ 3. results
R = ss.fa_res
c, seg, th = R.counts, R.segments, R.thresholds
floor = th["rarity_floor"]
nseg = c["segments"]
if c["seg_warning"]:
    tone, title = 2, f"경고 {c['seg_warning']}개 조각" + (f" · 주의 {c['seg_attention']}개 조각" if c["seg_attention"] else "")
elif c["seg_attention"]:
    tone, title = 1, f"주의 {c['seg_attention']}개 조각 · 경고 없음"
else:
    tone, title = 0, "이상 신호 없음"
desc = (f"{ui.esc(ss.fa_name)} · 전체 {c['rows_total']:,}행 중 {c['rows_analyzed']:,}행 분석"
        + (f" (중복 {c['rows_duplicate']}행은 원래 행 판정을 따름, 제외 {c['rows_invalid']}행)" if c["rows_duplicate"] or c["rows_invalid"] else "")
        + f". 조각 {nseg:,}개를 판정했습니다"
        + (f"(그중 {c['segments_short']}개는 {MIN_ROWS}행 미만이라 가장 가까운 조각의 판정을 따름)." if c["segments_short"] else "."))
st.markdown(ui.band(title, desc, tone, [
    ("전체 행", f"{c['rows_total']:,}", f"분석 {c['rows_analyzed']:,}"),
    ("정상 조각", f"{c['seg_normal']:,}", f"{c['rows_normal']:,}행"),
    ("주의 조각", f"{c['seg_attention']:,}", f"{c['rows_attention']:,}행"),
    ("경고 조각", f"{c['seg_warning']:,}", f"{c['rows_warning']:,}행"),
    ("경고 비율", f"{c['seg_warning'] / nseg:.1%}", f"행 기준 {c['rows_warning'] / max(c['rows_total'] - c['rows_invalid'], 1):.1%}"),
]), unsafe_allow_html=True)
st.markdown(f'<div class="note" style="margin-top:8px">판정 기준: 학습에 쓰지 않은 정상 조각 점수와 비교한 희귀도가 상위 '
            f'{100 * (1 - th["attention"]):.2f}% 이내면 주의, {100 * (1 - th["warning"]):.2f}% 이내면 경고입니다. '
            f'희귀도는 고장 확률이 아니라 "정상 기록 중 이보다 드문 비율"입니다. 기준이 정상 점수의 상위 약 5%·1%이므로 '
            f'정상 운전 기록에서도 조각의 약 5%는 주의, 약 1%는 경고가 나오는 것이 설계상 정상입니다(교차검증 정상 오경보 1.3%). '
            f'판정은 조각 단위이고 한 조각의 모든 행이 같은 판정을 받습니다. 원인을 확정하지 않는 이상 신호 후보로 해석하세요.</div>',
            unsafe_allow_html=True)
b1, b2, _ = st.columns([1.2, 1.4, 4])
b1.button("다른 파일 분석", on_click=reset, width="stretch")
P = R.rows.rename(columns={"rarity": "rarity_1_minus_p"})
b2.download_button("행별 결과 CSV", P.to_csv(encoding="utf-8-sig").encode("utf-8-sig"),
                   file_name=f"predictions_{os.path.splitext(ss.fa_name)[0]}.csv", mime="text/csv", width="stretch")

# ---- chart
st.markdown("### 신호와 판정")
top = st.columns([3, 2])
with top[0]:
    st.markdown(ui.legend([("정상", ui.LEVEL_COL[0]), ("주의", ui.LEVEL_COL[1]), ("경고", ui.LEVEL_COL[2])])
                + '<p class="note">선은 연속 구간 안에서만 이어집니다(끊긴 곳은 실제 측정 공백). 속이 빈 점은 8행 미만이라 '
                  '이웃 조각의 판정을 따른 조각입니다. 드래그로 확대, 더블클릭으로 원래대로.</p>', unsafe_allow_html=True)
with top[1]:
    xmode = st.segmented_control("가로축", ["측정 시각", "원본 index"], default="측정 시각", key="fa_xmode")
ranked = seg.sort_values(["level", "rarity", "d2_joint"], ascending=[False, True, False])
alerts = ranked[ranked.level >= 1]
focus_id = ss.get("fa_focus")
if focus_id not in set(seg.segment_id):
    focus_id = int(alerts.segment_id.iloc[0]) if len(alerts) else None
st.plotly_chart(timeline(ss.fa_hash, "index" if xmode == "원본 index" else "time", focus_id, R), width="stretch",
                config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})

# ---- anomaly list
st.markdown("### 이상 조각")


def table(df):
    return pd.DataFrame({
        "조각": df.segment_id.values,
        "원본 index": [f"{a}–{b}" for a, b in zip(df.start_index, df.end_index)],
        "시작 시각": pd.to_datetime(df.start_time).dt.strftime("%H:%M:%S").values,
        "행 수": df.rows.values,
        "판정": df.level_name.values,
        "희귀도(정상 대비 상위)": [ui.fmt_rarity(r, floor) for r in df.rarity],
        "이상 거리(통합)": [("-" if np.isnan(v) else f"{v:,.1f}") for v in df.d2_joint],
        "신호 유형": df.diagnosis.values,
        "주요 특징": df.top_features.values,
        "판정 근거": df.reason.values,
    })


if len(alerts):
    st.caption(f"경고·주의 {len(alerts)}개를 단계 → 희귀도 → 이상 거리 순으로 정렬했습니다. 이상 거리는 정상 조각의 99% 지점이 "
               f"{th['d2_joint_q99']:.1f}입니다. 행을 선택하면 아래에 상세가 표시됩니다.")
    shown = alerts
else:
    st.markdown('<div class="panel ok"><div class="t">주의·경고로 판정된 조각이 없습니다</div><div class="note">모든 조각이 정상 기록의 '
                f'상위 {100 * (1 - th["attention"]):.2f}%보다 흔한 범위에 있습니다. 참고로 희귀도가 가장 높은 조각 5개를 보여 줍니다.</div></div>',
                unsafe_allow_html=True)
    shown = seg[seg.judged_directly].sort_values("rarity").head(5)
ev = st.dataframe(table(shown), hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                  key="fa_table", height=min(38 + 35 * len(shown), 420))
sel = ev.selection.rows if ev is not None else []
if sel:
    focus_id = int(shown.segment_id.iloc[sel[0]])
    if ss.get("fa_focus") != focus_id:
        ss.fa_focus = focus_id
        st.rerun()
elif focus_id is None and len(shown):
    focus_id = int(shown.segment_id.iloc[0])

# ---- segment detail
if focus_id is not None:
    f = seg[seg.segment_id == focus_id].iloc[0]
    st.markdown(f"### 조각 {focus_id} 상세")
    l, r = st.columns([3, 2], gap="large")
    with l:
        st.plotly_chart(ui.segment_fig(R.rows, f, seg), width="stretch", config={"displaylogo": False})
    with r:
        kv = [("판정", f"{f.level_name}" + ("" if f.judged_directly else " (이웃 조각 판정 따름)")),
              ("원본 index", f"{f.start_index} – {f.end_index} ({f.rows}행)"),
              ("측정 시각", f"{pd.to_datetime(f.start_time):%H:%M:%S.%f}"[:-3] + f" ~ {pd.to_datetime(f.end_time):%H:%M:%S.%f}"[:-3]),
              ("희귀도", ui.fmt_rarity(f.rarity, floor) + " (정상 대비 상위)"),
              ("관점별", "-" if np.isnan(f.p_joint) else f"통합 {ui.fmt_rarity(1 - f.p_joint, floor)} · 전류 전용 {ui.fmt_rarity(1 - f.p_current, floor)}"),
              ("이상 거리", "-" if np.isnan(f.d2_joint) else f"통합 {f.d2_joint:,.1f} (정상 99% 지점 {th['d2_joint_q99']:.1f})"),
              ("추세 경보", "있음" if f.trend_warning else "없음"),
              ("신호 유형", f.diagnosis or "-")]
        st.markdown('<div class="kv">' + "".join(f'<span class="k">{ui.esc(k)}</span><span>{ui.esc(v)}</span>' for k, v in kv)
                    + "</div>", unsafe_allow_html=True)
        if f.judged_directly:
            x = R.rows[(R.rows.row_status == "분석") & (R.rows.segment_id == focus_id)].sort_values("TimeStamp")[
                ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]].to_numpy(float)
            cc = B["system"].det["joint"].window_contributions(x)
            if cc is not None:
                sh = np.clip(cc.mean(0), 0, None)
                sh = sh / sh.sum() if sh.sum() > 0 else sh
                st.markdown('<p class="note" style="margin:12px 0 0 0">통합 관점 점수에서 각 특징이 차지하는 비중</p>', unsafe_allow_html=True)
                st.plotly_chart(ui.contrib_fig(sh), width="stretch", config={"displayModeBar": False})
        else:
            st.markdown(f'<p class="note" style="margin-top:12px">{f.rows}행 조각이라 특징을 계산할 수 없어 직접 판정하지 않았습니다. '
                        "시간상 가장 가까운 판정 조각의 결과를 따릅니다.</p>", unsafe_allow_html=True)

# ---- row data
with st.expander("행별 데이터 보기"):
    flt = st.segmented_control("표시", ["전체", "경고", "주의", "정상", "중복 행", "제외"], default="전체", key="fa_rowfilter")
    V = R.rows
    if flt in ("경고", "주의", "정상"):
        V = V[(V.level_name == flt) & (V.row_status != "제외")]
    elif flt in ("중복 행", "제외"):
        V = V[V.row_status == flt]
    cols = [c for c in ["TimeStamp", "AI0_Vibration", "AI1_Vibration", "AI2_Current", "row_status", "segment_id", "level_name",
                        "rarity", "judged_directly", "diagnosis"] if c in V]
    nseg_v = V.segment_id.dropna().nunique()
    st.caption(f"{len(V):,}행 · {nseg_v:,}개 조각 · 판정은 조각 단위라 한 조각의 행은 모두 같은 판정입니다. index는 원본 파일의 행 번호입니다.")
    st.dataframe(V[cols].rename(columns={"row_status": "상태", "segment_id": "조각", "level_name": "판정", "rarity": "희귀도(0~1)",
                                         "judged_directly": "직접 판정", "diagnosis": "신호 유형"}),
                 width="stretch", height=360)
