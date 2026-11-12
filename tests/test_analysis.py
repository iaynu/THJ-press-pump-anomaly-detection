import os
import pickle

import pandas as pd
import pytest

from kamp.analysis import analyze, read_csv, validate, InputError

MODEL = "models/final_system.pkl"


def test_missing_column_is_reported():
    d = pd.DataFrame({"TimeStamp": ["2024-01-01 00:00:00"], "AI0_Vibration": [0.0]})
    with pytest.raises(InputError, match="필수 열"):
        validate(d)


def test_too_few_rows_is_reported():
    d = pd.DataFrame({"TimeStamp": pd.date_range("2024-01-01", periods=3, freq="100ms"),
                      "AI0_Vibration": [0.0] * 3, "AI1_Vibration": [0.0] * 3, "AI2_Current": [0.0] * 3})
    with pytest.raises(InputError):
        validate(d)


@pytest.fixture(scope="module")
def bundle():
    if not os.path.exists(MODEL):
        pytest.skip("models/final_system.pkl 없음 (python run.py train)")
    with open(MODEL, "rb") as f:
        return pickle.load(f)


def test_outlier_file_all_segments_warn_and_all_rows_kept(bundle):
    d = read_csv("data/outlier_data.csv")
    R = analyze(d, bundle)
    assert len(R.rows) == len(d)
    assert R.counts["segments"] == 21 and R.counts["seg_warning"] == 21


def test_normal_file_warning_rate_is_low(bundle):
    d = read_csv("data/press_data_normal.csv")
    R = analyze(d, bundle)
    assert len(R.rows) == len(d)
    assert R.counts["rows_duplicate"] == 1
    assert R.counts["seg_warning"] / R.counts["segments"] < 0.02
