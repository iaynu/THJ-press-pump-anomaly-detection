"""Press hydraulic pump anomaly detection - Streamlit app.

    streamlit run app.py

Two pages:
  파일 분석        upload a CSV (or use a provided file) -> check -> analyse -> results, anomaly list, segment detail
  모니터링 시연     replay held-out normal data, the real anomalous recording and injected faults segment by segment,
                  with the 2-stage alarm and the simulated stop request
"""
import os
import sys

import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)
from kamp import ui  # noqa: E402

st.set_page_config(page_title="프레스 유압펌프 이상탐지", layout="wide", page_icon=":material/monitoring:",
                   initial_sidebar_state="collapsed")
st.markdown(ui.CSS, unsafe_allow_html=True)

pages = [st.Page("views/analysis.py", title="파일 분석", icon=":material/upload_file:", default=True, url_path="analysis"),
         st.Page("views/monitor.py", title="모니터링 시연", icon=":material/play_circle:", url_path="monitor")]
st.navigation(pages, position="top").run()
