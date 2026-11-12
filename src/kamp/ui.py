"""Shared look and charts for the Streamlit app (app.py, views/).

One small design system: neutral surfaces, one accent for actions, and colour reserved for alarm state
(normal / attention / warning / not judged). Charts never compress time: every sample is drawn at its real
0.1 s spacing (or at its original row index), so a short segment looks short and a gap looks like a gap.
"""
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

from .features import FEATURES

INK, MUTED, LINE, SURFACE = "#1d232b", "#5d6773", "#e3e6ea", "#ffffff"
ACCENT = "#1f5f6b"
LEVEL_COL = {0: "#5b6b7b", 1: "#c27c0e", 2: "#c2362b"}
HOLD = "#a3abb5"
LEVEL_NAME = {0: "정상", 1: "주의", 2: "경고"}
CHANNELS = ["상부 진동", "하부 진동", "전류"]
FONT = "IBM Plex Sans KR, Pretendard, Apple SD Gothic Neo, Malgun Gothic, sans-serif"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stDataFrame, button, input, textarea, select {font-family: %(font)s;}
.block-container, [data-testid="stMainBlockContainer"] {max-width: 1240px; padding-top: 4.6rem !important; padding-bottom: 4rem;}
h1, h2, h3 {letter-spacing: -0.01em; color: %(ink)s;}
h1 {font-size: 1.35rem !important; font-weight: 700 !important; padding: 0 !important;}
h3 {font-size: 1.02rem !important; font-weight: 600 !important; padding: 1.2rem 0 0.2rem 0 !important;}
[data-testid="stCaptionContainer"] {color: %(muted)s;}
.app-head {display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; margin-bottom: .2rem;}
.app-head .name {font-size: 1.05rem; font-weight: 700; color: %(ink)s;}
.app-head .sub {font-size: .85rem; color: %(muted)s;}
/* stepper */
.steps {display:flex; gap:0; margin: .6rem 0 1.1rem 0; border-bottom: 1px solid %(line)s; flex-wrap: wrap;}
.steps .s {padding: .55rem 1.1rem .55rem 0; font-size: .86rem; color: %(muted)s; display:flex; align-items:center; gap:.45rem; margin-right: 1rem;}
.steps .s b {display:inline-flex; width:20px; height:20px; border-radius:50%%; align-items:center; justify-content:center;
             font-size:.72rem; border:1px solid %(line)s; color:%(muted)s; font-weight:600;}
.steps .s.done b {background:%(ink)s; color:#fff; border-color:%(ink)s;}
.steps .s.now {color:%(ink)s; font-weight:600; box-shadow: inset 0 -2px 0 %(accent)s;}
.steps .s.now b {border-color:%(accent)s; color:%(accent)s;}
/* summary band: one surface, columns separated by rules */
.band {display:flex; flex-wrap:wrap; border:1px solid %(line)s; border-radius:8px; background:%(surface)s; overflow:hidden;}
.band .verdict {flex: 1 1 260px; padding: 14px 18px; border-right:1px solid %(line)s;}
.band .verdict .t {font-size: 1.05rem; font-weight: 700;}
.band .verdict .d {font-size: .84rem; color:%(muted)s; margin-top: 4px; line-height:1.5;}
.band .m {flex: 1 1 120px; padding: 14px 16px; border-right:1px solid %(line)s;}
.band .m:last-child {border-right: none;}
.band .m .k {font-size: .76rem; color:%(muted)s;}
.band .m .v {font-size: 1.35rem; font-weight: 700; font-variant-numeric: tabular-nums; color:%(ink)s; line-height:1.35;}
.band .m .s {font-size: .74rem; color:%(muted)s;}
@media (max-width: 760px) { .band .verdict {flex-basis:100%%; border-right:none; border-bottom:1px solid %(line)s;}
  .band .m {flex-basis: 45%%; border-bottom:1px solid %(line)s;} }
.dot {display:inline-block; width:9px; height:9px; border-radius:50%%; margin-right:6px; vertical-align:baseline;}
.legend {display:flex; gap:16px; flex-wrap:wrap; font-size:.8rem; color:%(muted)s; margin:.1rem 0 .3rem 0;}
.note, [data-testid="stMarkdownContainer"] p.note, [data-testid="stMarkdownContainer"] .note {font-size:.84rem !important; color:%(muted)s; line-height:1.55;}
.panel {border:1px solid %(line)s; border-radius:8px; background:%(surface)s; padding:16px 18px;}
.panel.err {border-color:#e8c4c0; background:#fdf6f5;}
.panel.ok {border-color:#cfe0d6; background:#f5faf7;}
.panel .t {font-weight:600; margin-bottom:4px;}
.kv {display:grid; grid-template-columns: max-content 1fr; gap: 4px 16px; font-size:.86rem;}
.kv .k {color:%(muted)s;}
.stop {border:1px solid #e5b9b4; background:#fbf1f0; border-left:4px solid %(warn)s; border-radius:6px; padding:10px 14px; margin:.4rem 0 .2rem 0;}
.stop .t {font-weight:700; color:#8f241b;}
div[data-testid="stDataFrame"] {border:1px solid %(line)s; border-radius:6px;}
</style>
""" % dict(font=FONT, ink=INK, muted=MUTED, line=LINE, surface=SURFACE, accent=ACCENT, warn=LEVEL_COL[2])


def esc(s):
    return html.escape(str(s))


def head(name, sub):
    return f'<div class="app-head"><span class="name">{esc(name)}</span><span class="sub">{esc(sub)}</span></div>'


def stepper(labels, current, done_upto=None):
    done_upto = current - 1 if done_upto is None else done_upto
    out = []
    for i, lab in enumerate(labels):
        cls = "now" if i == current else ("done" if i <= done_upto else "")
        mark = "✓" if i <= done_upto and i != current else str(i + 1)
        out.append(f'<div class="s {cls}"><b>{mark}</b>{esc(lab)}</div>')
    return '<div class="steps">' + "".join(out) + "</div>"


def band(verdict_title, verdict_desc, tone, metrics):
    """tone: 0 normal / 1 attention / 2 warning / None neutral; metrics: [(label, value, sub)]"""
    col = INK if tone is None else (LEVEL_COL[tone] if tone else "#2f6b4f")
    ms = "".join(f'<div class="m"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div><div class="s">{esc(s)}</div></div>'
                 for k, v, s in metrics)
    return (f'<div class="band"><div class="verdict"><div class="t" style="color:{col}">{esc(verdict_title)}</div>'
            f'<div class="d">{verdict_desc}</div></div>{ms}</div>')


def legend(items):
    return '<div class="legend">' + "".join(f'<span><span class="dot" style="background:{c}"></span>{esc(t)}</span>'
                                            for t, c in items) + "</div>"


def fmt_rarity(r, floor):
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return "-"
    if r <= floor * 1.0001:
        return f"< {100 * floor:.2f}%"
    return f"{100 * r:.2f}%" if r < 0.1 else f"{100 * r:.1f}%"


def base_layout(fig, height, **kw):
    fig.update_layout(height=height, template="plotly_white", font=dict(family=FONT, size=12, color=INK),
                      margin=dict(l=8, r=12, t=28, b=8), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                      hoverlabel=dict(font=dict(family=FONT, size=12), bgcolor="white", bordercolor=LINE),
                      showlegend=False, **kw)
    fig.update_xaxes(gridcolor="#eef0f3", zeroline=False, linecolor=LINE, ticks="outside", tickcolor=LINE)
    fig.update_yaxes(gridcolor="#eef0f3", zeroline=False, linecolor=LINE, title_font=dict(size=11, color=MUTED))
    return fig


# ------------------------------------------------------------------------------------------------ rarity axis
# rarity (share of normal segments that are even more unusual) is drawn on a linear axis of -log10(rarity):
# up = rarer. Ticks are labelled in %. (A plotly log axis misplaces annotations, so the transform is done here.)
RARITY_TICKS = [100, 30, 10, 3, 1, 0.3]


def rv(pct):
    """rarity in % -> plotting height"""
    return -np.log10(np.asarray(pct, dtype=float) / 100)


def _rarity_axis(fig, row=None, col=None):
    kw = dict(row=row, col=col) if row else {}
    fig.update_yaxes(range=[-0.08, float(rv(0.13))], tickvals=list(rv(RARITY_TICKS)), ticktext=[f"{v:g}%" for v in RARITY_TICKS], **kw)


def _threshold_lines(fig, th, row=None, col=None, labels=True):
    kw = dict(row=row, col=col) if row else {}
    for p, lab, c, pos in [(th["attention"], "주의", LEVEL_COL[1], "bottom right"), (th["warning"], "경고", LEVEL_COL[2], "top right")]:
        v = 100 * (1 - p)
        fig.add_hline(y=float(rv(v)), line=dict(color=c, dash="dash", width=1),
                      annotation_text=f"{lab} 기준 상위 {v:.2f}%" if labels else None, annotation_position=pos,
                      annotation_font=dict(size=10, color=c), annotation_bgcolor="rgba(255,255,255,0.85)", **kw)


# ------------------------------------------------------------------------------------------------ file analysis
def _runs(df, xcol):
    """arrays for one trace: samples in time order with a NaN after every segment, so lines never bridge a gap"""
    seg = df["segment_id"].to_numpy()
    brk = np.r_[np.nonzero(seg[1:] != seg[:-1])[0] + 1, len(seg)]          # position after each segment
    ins = lambda a, fill: np.insert(np.asarray(a, dtype=object), brk, fill)  # noqa: E731
    xs = ins(df[xcol].to_numpy(), None)
    ys = [ins(df[c].to_numpy(float), None) for c in ("AI0_Vibration", "AI1_Vibration", "AI2_Current")]
    cd = np.column_stack([ins(df.index.to_numpy(), None), ins(seg, None)])
    return xs, ys, cd


def file_timeline(rows, segs, th, xmode="time", focus=None):
    """4 stacked panels on one x axis: segment rarity, upper / lower vibration, current.
    rows: analysis rows (only row_status == '분석' are drawn; duplicates would be drawn twice)."""
    d = rows[rows.row_status == "분석"].copy()
    d = d.sort_values("TimeStamp", kind="stable")
    d["x"] = pd.to_datetime(d["TimeStamp"]) if xmode == "time" else d.index
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.035, row_heights=[0.22, 0.26, 0.26, 0.26])
    short = set(segs.loc[~segs.judged_directly, "segment_id"])
    xh = "%{x|%H:%M:%S.%L}" if xmode == "time" else "%{x}"
    for lv in (0, 1, 2):
        for is_short in (False, True):
            g = d[(d.level == lv) & (d.segment_id.isin(short) == is_short)]
            if g.empty:
                continue
            xs, ys, cd = _runs(g, "x")
            for k in range(3):
                fig.add_trace(go.Scattergl(
                    x=xs, y=ys[k], customdata=cd, mode="lines+markers" if is_short else "lines", connectgaps=False,
                    line=dict(color=LEVEL_COL[lv], width=1.1 if lv == 0 else 1.5),
                    marker=dict(size=5, symbol="circle-open", color=LEVEL_COL[lv]),
                    hovertemplate=(f"원본 index %{{customdata[0]}} · 조각 %{{customdata[1]}} · {LEVEL_NAME[lv]}"
                                   + ("" if not is_short else " (이웃 조각 판정 따름)")
                                   + f"<br>{xh} · {CHANNELS[k]} %{{y:.4f}}<extra></extra>"),
                    name=LEVEL_NAME[lv]), row=k + 2, col=1)
    # segment rarity: one point per segment at its middle, hollow if the segment was too short to judge itself
    s = segs.copy()
    if xmode == "time":
        s["xm"] = pd.to_datetime(s.start_time) + (pd.to_datetime(s.end_time) - pd.to_datetime(s.start_time)) / 2
    else:
        s["xm"] = (s.start_index + s.end_index) / 2
    s["y"] = rv(100 * np.maximum(s.rarity, th["rarity_floor"]))
    for lv in (0, 1, 2):
        for j in (True, False):
            g = s[(s.level == lv) & (s.judged_directly == j)]
            if g.empty:
                continue
            fig.add_trace(go.Scatter(
                x=g.xm, y=g.y, mode="markers",
                marker=dict(size=7 if lv else 5, color=LEVEL_COL[lv] if j else "white", line=dict(color=LEVEL_COL[lv], width=1.2)),
                customdata=np.column_stack([g.segment_id, g.start_index, g.end_index, g.rows,
                                            [fmt_rarity(r, th["rarity_floor"]) for r in g.rarity],
                                            np.where(g.judged_directly, "직접 판정", "8행 미만: 이웃 조각 판정 따름")]),
                hovertemplate=("조각 %{customdata[0]} · 원본 index %{customdata[1]}–%{customdata[2]} (%{customdata[3]}행)"
                               "<br>정상 대비 상위 %{customdata[4]} · " + LEVEL_NAME[lv] + "<br>%{customdata[5]}<extra></extra>")),
                row=1, col=1)
    _rarity_axis(fig, 1, 1)
    _threshold_lines(fig, th, 1, 1)
    if focus is not None:
        f = segs[segs.segment_id == focus].iloc[0]
        x0, x1 = ((pd.to_datetime(f.start_time), pd.to_datetime(f.end_time)) if xmode == "time"
                  else (f.start_index - 0.5, f.end_index + 0.5))
        for r in range(1, 5):
            fig.add_vrect(x0=x0, x1=x1, fillcolor=ACCENT, opacity=0.10, line_width=0, row=r, col=1)
    for r, t in enumerate(["희귀도 (정상 대비 상위 %)", "상부 진동", "하부 진동", "전류"], start=1):
        fig.update_yaxes(title_text=t, row=r, col=1)
    fig.update_xaxes(title_text="측정 시각" if xmode == "time" else "원본 index (파일의 행 번호)", row=4, col=1)
    base_layout(fig, 640, hovermode="closest")
    fig.update_layout(dragmode="zoom")
    return fig


def segment_fig(rows, seg, segs, context=1):
    """the selected segment's raw samples by original row index, with the neighbouring segments in grey"""
    order = segs.sort_values("start_time").segment_id.tolist()
    k = order.index(seg.segment_id)
    ids = order[max(0, k - context): k + context + 1]
    d = rows[(rows.row_status == "분석") & rows.segment_id.isin(ids)].sort_values("TimeStamp", kind="stable")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    for sid, g in d.groupby("segment_id", sort=False):
        me = sid == seg.segment_id
        col = LEVEL_COL[int(seg.level)] if me else HOLD
        for c, ch in enumerate(["AI0_Vibration", "AI1_Vibration", "AI2_Current"]):
            fig.add_trace(go.Scatter(x=g.index, y=g[ch], mode="lines+markers", line=dict(color=col, width=1.6 if me else 1),
                                     marker=dict(size=4 if me else 3, color=col),
                                     hovertemplate=f"원본 index %{{x}} · 조각 {sid}<br>{CHANNELS[c]} %{{y:.4f}}<extra></extra>"),
                          row=c + 1, col=1)
    for r, t in enumerate(CHANNELS, start=1):
        fig.update_yaxes(title_text=t, row=r, col=1)
    fig.update_xaxes(title_text="원본 index (회색: 앞뒤 조각)", row=3, col=1)
    return base_layout(fig, 420)


def contrib_fig(shares, height=260):
    order = np.argsort(shares)
    fig = go.Figure(go.Bar(x=shares[order], y=[FEATURES[i] for i in order], orientation="h",
                           marker_color=[ACCENT if shares[i] == shares.max() else "#b9c4cf" for i in order],
                           hovertemplate="%{y}: %{x:.0%}<extra></extra>"))
    fig.update_xaxes(tickformat=".0%", range=[0, max(0.5, float(shares.max()) * 1.12)], title_text="점수에서 차지하는 비중")
    return base_layout(fig, height)


# ------------------------------------------------------------------------------------------------ live monitor
GAP_VIS = 0.6   # seconds drawn between two segments (the replay joins recordings, so real gaps are not meaningful)


def stream_positions(steps):
    """start time (s) of each step on the replay axis: samples keep their real 0.1 s spacing"""
    t, out = 0.0, []
    for s in steps:
        out.append(t)
        t += 0.1 * (len(s.x) - 1) + GAP_VIS
    return np.array(out)


def monitor_signal(steps, results, starts, onset=None, window_s=40.0, hover=False):
    n = len(results)
    end = starts[n - 1] + 0.1 * (len(steps[n - 1].x) - 1)
    lo_t = max(0.0, end - window_s)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    first = next((i for i in range(n) if starts[i] + 0.1 * len(steps[i].x) >= lo_t), 0)
    for i in range(first, n):
        r, x = results[i], steps[i].x
        col = LEVEL_COL[r.level] if r.judged else HOLD
        t = starts[i] + 0.1 * np.arange(len(x))
        status = LEVEL_NAME[r.level] if r.judged else "판정 보류(8행 미만)"
        for k in range(3):
            fig.add_trace(go.Scatter(x=t, y=x[:, k], mode="lines+markers" if len(x) < 8 else "lines",
                                     line=dict(color=col, width=1.8 if i == n - 1 else 1.2, dash=None if r.judged else "dot"),
                                     marker=dict(size=4, color=col),
                                     hovertemplate=f"조각 {i + 1} ({len(x)}행) · {status}<br>{CHANNELS[k]} %{{y:.4f}}<extra></extra>"),
                          row=k + 1, col=1)
    # segment bands: every other segment gets a light background so boundaries are visible on a plain time axis;
    # the segment number sits above its band when the segment is wide enough to hold the label
    for j, i in enumerate(range(first, n)):
        x0, x1 = starts[i] - GAP_VIS / 2, starts[i] + 0.1 * (len(steps[i].x) - 1) + GAP_VIS / 2
        if (i % 2) == 0:
            fig.add_vrect(x0=x0, x1=x1, fillcolor="#eef1f5", opacity=1, line_width=0, layer="below", row="all", col=1)
        if x1 - x0 >= 1.6 and x1 > lo_t:
            fig.add_annotation(x=(x0 + x1) / 2, y=1.0, yref="paper", text=f"#{i + 1}", showarrow=False, yanchor="bottom",
                               font=dict(size=10, color=MUTED))
    if onset is not None and onset < n:
        fig.add_vline(x=starts[onset] - GAP_VIS / 2, line=dict(color=INK, dash="dot", width=1.2), row="all", col=1)
        fig.add_annotation(x=starts[onset] - GAP_VIS / 2, y=1.07, yref="paper", text=f"이상 시작 ({onset + 1}번째 조각)",
                           showarrow=False, xanchor="left", yanchor="bottom", font=dict(size=10, color=INK))
    fig.update_xaxes(range=[lo_t - 0.3, max(end, lo_t + window_s) + 0.3], dtick=5, row=3, col=1,
                     title_text="재생 시간(초) · 0.1초 간격 그대로, 조각 사이 공백은 0.6초로 압축")
    for r, t in enumerate(CHANNELS, start=1):
        fig.update_yaxes(title_text=t, row=r, col=1)
    base_layout(fig, 430, hovermode="closest" if hover else False)
    fig.update_layout(margin=dict(l=8, r=12, t=44, b=8))
    return fig


def monitor_score(steps, results, th, trend, onset=None, stops=()):
    fig = go.Figure()
    floor = th["rarity_floor"]
    total = len(steps)
    ok = [r for r in results if r.judged]
    hold = [r for r in results if not r.judged]
    if onset is not None:
        fig.add_vrect(x0=onset + 0.5, x1=total + 0.5, fillcolor=INK, opacity=0.05, line_width=0)
        fig.add_annotation(x=total + 0.4, y=0.0, yref="y", text="이상 구간(시나리오)", showarrow=False, xanchor="right",
                           yanchor="bottom", font=dict(size=10, color=MUTED))
    _threshold_lines(fig, th)
    lim = 100 * norm.sf(trend["ewma_z"])
    fig.add_hline(y=float(rv(lim)), line=dict(color=MUTED, dash="dot", width=1), annotation_text=f"추세 한계 상위 {lim:.0f}%",
                  annotation_position="top left", annotation_font=dict(size=10, color=MUTED))
    if results:
        fig.add_trace(go.Scatter(x=[r.i + 1 for r in results], y=list(rv(100 * norm.sf([r.ewma for r in results]))), mode="lines",
                                 line=dict(color=MUTED, dash="dot", width=1.2), name="추세(EWMA)",
                                 customdata=[f"{100 * norm.sf(r.ewma):.1f}%" for r in results],
                                 hovertemplate="조각 %{x} · 추세 상위 %{customdata}<extra></extra>"))
    if ok:
        fig.add_trace(go.Scatter(x=[r.i + 1 for r in ok], y=list(rv([100 * max(1 - r.pct, floor) for r in ok])), mode="lines+markers",
                                 line=dict(color="#c9d0d8", width=1),
                                 marker=dict(size=8, color=[LEVEL_COL[r.level] for r in ok], line=dict(width=0)),
                                 customdata=[[LEVEL_NAME[r.level], fmt_rarity(1 - r.pct, floor)] for r in ok],
                                 hovertemplate="조각 %{x} · %{customdata[0]}<br>정상 대비 상위 %{customdata[1]}<extra></extra>"))
    if hold:
        fig.add_trace(go.Scatter(x=[r.i + 1 for r in hold], y=[0.0] * len(hold), mode="markers",
                                 marker=dict(size=8, symbol="circle-open", color=HOLD, line=dict(width=1.5)),
                                 hovertemplate="조각 %{x} · 8행 미만이라 판정 보류(이전 상태 유지)<extra></extra>"))
    for i in stops:
        fig.add_vline(x=i + 1, line=dict(color=LEVEL_COL[2], width=2))
        fig.add_annotation(x=i + 1, y=float(rv(8)), yref="y", text="정지 요청", showarrow=False, xanchor="right", yanchor="middle",
                           bgcolor="rgba(255,255,255,0.9)",
                           font=dict(size=10, color=LEVEL_COL[2]))
    fig.update_xaxes(range=[0.5, total + 0.5], title_text="조각 번호")
    fig.update_yaxes(title_text="희귀도 (정상 대비 상위 %)")
    _rarity_axis(fig)
    return base_layout(fig, 430)
