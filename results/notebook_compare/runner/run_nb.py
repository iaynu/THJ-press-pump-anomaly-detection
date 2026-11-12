import matplotlib
matplotlib.use('Agg')

# ======== CELL 0 [markdown] ========
# # KAMP Final Competition Pipeline — Google Colab
# 
# **Research question:** Can a normal-reference anomaly detector distinguish the provided
# abnormal-state recording while remaining robust to operating-regime and acquisition changes?
# 
# This notebook uses **normal TRAIN only** for scaling, regimes and model fitting, a chronological
# NORMAL CALIBRATION subset for cutoffs, and later normal data for evaluation. The abnormal file is
# retrospective evaluation only. It does not establish remaining useful life, failure probability,
# mechanical fault cause, or advance failure prediction. Sensor units and acquisition bandwidth are unverified.
# 
# **Use:** upload this one notebook to Colab; edit section 00's project root and section 02's explicit
# filenames; place CSVs in `<PROJECT_ROOT>/data`; run all cells. No repository imports are used.
# Every numbered section has editable configuration and visible analysis. Rerun configuration and
# its downstream cells after an edit. Cache reuse is checked against input hashes, configuration,
# package versions and generating functions. `FORCE_RECOMPUTE=True` bypasses caches.
# 
# Outputs remain in Drive after the Colab runtime ends. Manual upload is an alternative input route.
# The default submission is an **editable demonstration schema**, not a verified competition format.
# Sections 17–18 are recorded-data results; sections 20–22 are controlled sensitivity experiments.
# 
# References: [Colab file/Drive examples](https://colab.research.google.com/notebooks/io.ipynb),
# [MinCovDet](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.MinCovDet.html),
# [LedoitWolf](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html).
# MCD estimates a robust elliptical reference; multiple operating regimes can violate that assumption.
# Distances and calibration percentiles are **deviation evidence, not fault probabilities**.
# 

# ======== CELL 1 [markdown] ========
# ## 00. Environment & Configuration
# 
# Install only missing packages. Seeds, package versions and explicit assumptions are recorded.
# 

# ======== CELL 2 [code] ========
PROJECT_ROOT = __import__("os").environ.get("NB_PROJECT_ROOT", "/tmp/nbrun/proj")  # change once here (or set NB_PROJECT_ROOT)
RANDOM_STATE = 42
FORCE_RECOMPUTE = False
PIPELINE_REVISION = "colab-normal-reference-v1"
FIGURE_DPI = 160


# ======== CELL 3 [code] ========
import importlib.util, subprocess, sys
required_packages = {"numpy":"numpy", "pandas":"pandas", "scipy":"scipy",
                     "sklearn":"scikit-learn", "matplotlib":"matplotlib", "joblib":"joblib"}
missing_packages = [package for module, package in required_packages.items()
                    if importlib.util.find_spec(module) is None]
if missing_packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
import os, json, hashlib, marshal, warnings, platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy, sklearn, matplotlib, joblib
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, average_precision_score, precision_recall_fscore_support
from sklearn.covariance import MinCovDet, LedoitWolf
from sklearn.neighbors import NearestNeighbors
try:
    from IPython.display import display, Markdown
except ImportError:
    def display(value): print(value)
    def Markdown(value): return value
np.random.seed(RANDOM_STATE)
VERSIONS = {name: getattr(module, "__version__", "unknown") for name, module in
            [("numpy",np), ("pandas",pd), ("scipy",scipy), ("sklearn",sklearn),
             ("matplotlib",matplotlib), ("joblib",joblib)]}
VERSIONS["python"] = platform.python_version()
print("Versions:", VERSIONS)
plt.rcParams.update({"figure.figsize": (9, 4.8), "axes.grid": True, "grid.alpha": .2,
                     "font.size": 10, "savefig.bbox": "tight"})


# ======== CELL 4 [markdown] ========
# ## 01. Mount Google Drive
# 
# Drive is the primary workflow. For local verification the mount is skipped; all other code stays the same.
# 

# ======== CELL 5 [code] ========
USE_GOOGLE_DRIVE = False
MANUAL_UPLOAD = False


# ======== CELL 6 [code] ========
try:
    IN_COLAB = importlib.util.find_spec("google.colab") is not None
except ModuleNotFoundError:
    IN_COLAB = False
if USE_GOOGLE_DRIVE and IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
elif USE_GOOGLE_DRIVE:
    print("Outside Colab: set PROJECT_ROOT to a writable local directory before continuing.")
PROJECT_ROOT = Path(PROJECT_ROOT)
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = OUTPUT_DIR / "models"
CACHE_DIR = OUTPUT_DIR / "cache"
for directory in (DATA_DIR, OUTPUT_DIR, FIGURE_DIR, TABLE_DIR, MODEL_DIR, CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

def save_table(frame, name):
    frame.to_csv(TABLE_DIR / name, index=False, encoding="utf-8-sig")
    return frame

def save_figure(name):
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / name, dpi=FIGURE_DPI)
    plt.show()
    plt.close()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

def config_snapshot():
    # Uppercase configuration is explicit in every section, never inferred from performance.
    return {key: value for key, value in globals().items() if key.isupper()
            and isinstance(value, (str, int, float, bool, tuple, list, dict, Path, type(None)))
            and key not in ("RUN_CONFIG",)}

def cached_object(name, signature, producer, functions=()):
    function_hashes = [hashlib.sha256(marshal.dumps(f.__code__)).hexdigest() for f in functions]
    payload = {"revision":PIPELINE_REVISION, "versions":VERSIONS, "signature":signature,
               "functions":function_hashes}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    path, manifest = CACHE_DIR / f"{name}.joblib", CACHE_DIR / f"{name}.json"
    if not FORCE_RECOMPUTE and path.exists() and manifest.exists():
        if json.loads(manifest.read_text())["digest"] == digest:
            print(f"Cache hit: {name}")
            return joblib.load(path)  # only load artifacts generated by this notebook
    result = producer()
    joblib.dump(result, path)
    write_json(manifest, {"digest":digest, "payload":payload})
    print(f"Computed and saved: {name}")
    return result

print("Persistent output location:", OUTPUT_DIR)
write_json(OUTPUT_DIR / "run_config.json", config_snapshot())


# ======== CELL 7 [markdown] ========
# ## 02. Load Data
# 
# Names and columns are editable. Candidate CSVs are printed; no candidate is chosen silently. The optional sample submission is inspected in section 25.
# 

# ======== CELL 8 [code] ========
NORMAL_FILE = "press_data_normal.csv"  # relative to DATA_DIR, or absolute
ABNORMAL_FILE = "outlier_data.csv"
SAMPLE_SUBMISSION_FILE = None
SUBMISSION_INPUT_FILE = None  # None => export supplied abnormal recording; set for unlabeled evaluation CSV
TIMESTAMP_COLUMN = "TimeStamp"
SENSOR_COLUMNS = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
LABEL_COLUMN = "Equipment_state"  # never used as a model input
CSV_ENCODINGS = ["utf-8-sig", "cp949", "utf-16"]


# ======== CELL 9 [code] ========
if MANUAL_UPLOAD:
    from google.colab import files
    uploaded = files.upload()
    for filename, content in uploaded.items():
        (DATA_DIR / Path(filename).name).write_bytes(content)
    print("Uploaded files persist in:", DATA_DIR)
csv_candidates = sorted(str(path.relative_to(DATA_DIR)) for path in DATA_DIR.rglob("*.csv"))
print("CSV candidates (inspect and edit filenames above):", csv_candidates)

def resolve_input(filename):
    if filename is None: return None
    path = Path(filename)
    path = path if path.is_absolute() else DATA_DIR / path
    if not path.is_file(): raise FileNotFoundError(f"Set the explicit filename: {path}")
    return path

def read_sensor_csv(path, source, label):
    errors = []
    for encoding in CSV_ENCODINGS:
        try:
            frame = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeError as exc: errors.append(str(exc))
    else: raise ValueError(f"Cannot decode {path}: {errors}")
    required = [TIMESTAMP_COLUMN, *SENSOR_COLUMNS]
    if not set(required).issubset(frame.columns):
        raise ValueError(f"Missing columns; configure column mapping. Actual: {list(frame)}")
    frame = frame.copy()
    frame["source_row_id"] = np.arange(len(frame))  # original position; never reset on sort
    frame["source"] = source
    frame["evaluation_label"] = label
    return frame

normal_path, abnormal_path = resolve_input(NORMAL_FILE), resolve_input(ABNORMAL_FILE)
assert normal_path.resolve() != abnormal_path.resolve(), "Normal and abnormal files must differ"
normal_raw = read_sensor_csv(normal_path, "normal", 0)
abnormal_raw = read_sensor_csv(abnormal_path, "abnormal", 1)
INPUT_HASHES = {key:hashlib.sha256(path.read_bytes()).hexdigest()
                for key,path in [("normal",normal_path),("abnormal",abnormal_path)]}
print("Normal / abnormal raw rows:", len(normal_raw), len(abnormal_raw))
display(normal_raw.head())
print("Labels, if provided, are audit metadata; only the explicitly normal file is fitted.")


# ======== CELL 10 [markdown] ========
# ## 03. Data Integrity Audit
# 
# Audit before transformation. Exact duplicates are collapsed on all original CSV columns; different measurements at the same timestamp are retained and split at dt<=0. Original row-to-canonical mappings remain available. Invalid rows are retained for export/QC but excluded from model evidence.
# 

# ======== CELL 11 [code] ========
TIMESTAMP_PARSE_UTC = True  # parse uniformly; source timezone is unverified
EXPECTED_NORMAL_LABEL = 0
EXPECTED_ABNORMAL_LABEL = 1


# ======== CELL 12 [code] ========
def audit_and_clean(raw):
    original_columns = [c for c in raw.columns if c not in
                        ["source_row_id", "source", "evaluation_label"]]
    print("Source:", raw.source.iloc[0], "shape:", raw[original_columns].shape)
    print("Columns:", original_columns)
    display(raw[original_columns].dtypes.to_frame("dtype"))
    display(raw[original_columns].head())
    display(raw[original_columns].isna().sum().to_frame("missing"))
    parsed = pd.to_datetime(raw[TIMESTAMP_COLUMN], errors="coerce", format="mixed", utc=TIMESTAMP_PARSE_UTC)
    duplicate_mask = raw.duplicated(subset=original_columns, keep="first")
    # Factorization uses original CSV values, so timestamps with different measurements survive.
    canonical = raw.groupby(original_columns, dropna=False, sort=False)["source_row_id"].transform("min")
    mapping = raw[["source", "source_row_id", "evaluation_label"]].copy()
    mapping["canonical_row_id"] = canonical.astype(int)
    clean = raw.loc[~duplicate_mask].copy()
    clean["timestamp"] = parsed.loc[clean.index]
    for sensor in SENSOR_COLUMNS: clean[sensor] = pd.to_numeric(clean[sensor], errors="coerce")
    clean["valid_measurement"] = clean.timestamp.notna() & np.isfinite(clean[SENSOR_COLUMNS]).all(axis=1)
    conflicting = clean.loc[clean.timestamp.notna() & clean.timestamp.duplicated(keep=False)]
    audit = {"source":raw.source.iloc[0], "raw_rows":len(raw), "columns":str(original_columns),
             "exact_duplicate_rows":int(duplicate_mask.sum()),
             "duplicate_timestamp_rows":int(parsed.duplicated(keep=False).sum()),
             "conflicting_timestamp_rows_after_exact_dedup":len(conflicting),
             "invalid_timestamp_rows":int(parsed.isna().sum()),
             "invalid_model_rows":int((~clean.valid_measurement).sum()),
             "timestamp_min":parsed.min(), "timestamp_max":parsed.max(),
             "missing_values":int(raw[original_columns].isna().sum().sum())}
    display(clean[SENSOR_COLUMNS].describe())
    print("Audit:", audit)
    if len(conflicting): print("Conflicting timestamps preserved and isolated:", len(conflicting))
    if LABEL_COLUMN in raw:
        print("Supplied label distribution:", raw[LABEL_COLUMN].value_counts(dropna=False).to_dict())
        expected = EXPECTED_NORMAL_LABEL if raw.source.iloc[0] == "normal" else EXPECTED_ABNORMAL_LABEL
        if not raw[LABEL_COLUMN].eq(expected).all():
            raise ValueError("Unexpected source labels: review normal-reference eligibility before fitting")
    clean = clean.sort_values(["timestamp", "source_row_id"], kind="stable", na_position="last")
    return clean, mapping, audit, conflicting

normal_clean, normal_row_map, normal_audit, normal_conflicts = audit_and_clean(normal_raw)
abnormal_clean, abnormal_row_map, abnormal_audit, abnormal_conflicts = audit_and_clean(abnormal_raw)
data_audit = pd.DataFrame([normal_audit, abnormal_audit])
save_table(data_audit, "data_audit.csv")
save_table(pd.concat([normal_conflicts, abnormal_conflicts]), "duplicate_timestamp_audit.csv")
save_table(pd.concat([normal_row_map, abnormal_row_map]), "original_row_mapping.csv")
print("After exact dedup: normal", len(normal_clean), "abnormal", len(abnormal_clean))


# ======== CELL 13 [markdown] ========
# ## 04. Timestamp & Gap Analysis
# 
# Estimate dt from positive timestamp differences, and separately report original-order reversals. No interpolation bridges missing acquisition. Threshold comparison uses timestamps only, never abnormal recall.
# 

# ======== CELL 14 [code] ========
NOMINAL_DT_SEC = 0.1  # documented expectation, compared with observed median
NEAR_NOMINAL_TOLERANCE_SEC = 0.005
GAP_CANDIDATES_SEC = [0.15, 0.5, 1.0]


# ======== CELL 15 [code] ========
dt_records = []
for clean, raw in [(normal_clean,normal_raw),(abnormal_clean,abnormal_raw)]:
    dt = clean.timestamp.diff().dt.total_seconds()
    clean["dt_sec"] = dt
    positive = dt[dt > 0]
    raw_dt = pd.to_datetime(raw[TIMESTAMP_COLUMN], errors="coerce", format="mixed", utc=True).diff().dt.total_seconds()
    dt_records.append({"source":clean.source.iloc[0], "nominal_dt_sec":NOMINAL_DT_SEC,
                       "median_positive_dt_sec":positive.median(),
                       "mode_like_dt_sec":positive.round(3).mode().iloc[0] if len(positive) else np.nan,
                       "near_nominal_percent":100*dt.dropna().sub(NOMINAL_DT_SEC).abs().le(NEAR_NOMINAL_TOLERANCE_SEC).mean(),
                       "dt_le_0":int(dt.le(0).sum()), "raw_order_reversals":int(raw_dt.lt(0).sum()),
                       **{f"dt_gt_{threshold}":int(dt.gt(threshold).sum()) for threshold in GAP_CANDIDATES_SEC}})
timestamp_audit = pd.DataFrame(dt_records)
data_audit = data_audit.merge(timestamp_audit, on="source")
save_table(data_audit, "data_audit.csv")
display(timestamp_audit)
NORMAL_MEDIAN_DT = float(timestamp_audit.loc[timestamp_audit.source.eq("normal"), "median_positive_dt_sec"].iloc[0])
if not np.isfinite(NORMAL_MEDIAN_DT) or NORMAL_MEDIAN_DT <= 0: raise ValueError("Cannot infer normal sampling rate")
OBSERVED_FS_HZ = 1 / NORMAL_MEDIAN_DT
print(f"Normal observed median rate: {OBSERVED_FS_HZ:.3f} Hz; Nyquist {OBSERVED_FS_HZ/2:.3f} Hz")
for name, limits in [("dt_distribution.png",None),("dt_distribution_0_2sec.png",(0,2)),
                     ("large_gap_distribution.png",(.5,None))]:
    plt.figure()
    for frame in [normal_clean,abnormal_clean]:
        values = frame.dt_sec.dropna()
        if limits: values = values[values >= limits[0]]
        if limits and limits[1] is not None: values = values[values <= limits[1]]
        if len(values): plt.hist(values, bins=40, alpha=.45, density=True, label=frame.source.iloc[0])
    if limits and limits[1] is not None: plt.xlim(*limits)
    plt.xlabel("Timestamp difference (seconds)"); plt.ylabel("Relative density")
    plt.title("Acquisition timestamp intervals"); plt.legend(); save_figure(name)
plt.figure()
for frame in [normal_clean,abnormal_clean]:
    plt.step(frame.timestamp, np.arange(1,len(frame)+1), label=frame.source.iloc[0])
plt.xlabel("Actual timestamp (parsed UTC; source timezone unverified)"); plt.ylabel("Cumulative recorded samples")
plt.title("Recorded sample count over actual time"); plt.legend(); save_figure("samples_over_timestamp.png")


# ======== CELL 16 [markdown] ========
# ## 05. Continuous Segment Construction
# 
# Default gap >0.5s follows the requested structural rule. dt<=0 and invalid measurements also form boundaries. These are acquisition segments, not proven press cycles. FFT/shape additionally require near-uniform sampling inside each window.
# 

# ======== CELL 17 [code] ========
SEGMENT_GAP_SEC = 0.5
MIN_PHYSICS_SAMPLES = 8
UNIFORM_DT_REL_TOL = 0.10


# ======== CELL 18 [code] ========
def construct_segments(clean, gap_sec):
    frame = clean.copy()
    dt = frame.timestamp.diff().dt.total_seconds()
    invalid = ~frame.valid_measurement
    boundary = dt.isna() | dt.le(0) | dt.gt(gap_sec) | invalid | invalid.shift(fill_value=False)
    numbers = boundary.cumsum().astype(int)-1
    frame["segment_id"] = frame.source.iloc[0] + ":" + numbers.astype(str)
    summary = frame.groupby(["source","segment_id"], sort=False).agg(
        segment_start=("timestamp","min"), segment_end=("timestamp","max"),
        segment_n_samples=("source_row_id","size"), valid_segment=("valid_measurement","all"),
        gap_before_sec=("dt_sec","first"), evaluation_label=("evaluation_label","first")).reset_index()
    summary["segment_duration_sec"] = (summary.segment_end-summary.segment_start).dt.total_seconds()
    # duration excludes the acquisition gap before a segment.
    frame = frame.merge(summary.drop(columns=["source","evaluation_label","valid_segment"]), on="segment_id", validate="many_to_one")
    return frame, summary

structural_records = []
for gap in GAP_CANDIDATES_SEC:
    for clean in [normal_clean, abnormal_clean]:
        _, stats_frame = construct_segments(clean, gap)
        structural_records.append({"source":clean.source.iloc[0], "gap_sec":gap, "n_segments":len(stats_frame),
            "median_samples":stats_frame.segment_n_samples.median(), "min_samples":stats_frame.segment_n_samples.min(),
            "max_samples":stats_frame.segment_n_samples.max(), "median_duration_sec":stats_frame.segment_duration_sec.median(),
            "short_sample_percent":100*stats_frame.loc[stats_frame.segment_n_samples < MIN_PHYSICS_SAMPLES,"segment_n_samples"].sum()/len(clean)})
        plt.figure(); plt.hist(stats_frame.segment_n_samples, bins=25, color="steelblue")
        plt.xlabel("Samples per acquisition segment"); plt.ylabel("Segment count")
        plt.title(f"{clean.source.iloc[0]}: gap > {gap}s"); save_figure(f"segment_lengths_{clean.source.iloc[0]}_{gap}.png")
segment_structure = save_table(pd.DataFrame(structural_records), "segmentation_sensitivity.csv")
display(segment_structure)
normal, normal_segments = construct_segments(normal_clean, SEGMENT_GAP_SEC)
abnormal, abnormal_segments = construct_segments(abnormal_clean, SEGMENT_GAP_SEC)
segment_summary = pd.concat([normal_segments, abnormal_segments], ignore_index=True)
save_table(segment_summary, "segment_summary.csv")
for frame in [normal,abnormal]:
    plt.figure()
    for _, segment in frame.groupby("segment_id",sort=False):
        plt.plot(segment.timestamp, segment[SENSOR_COLUMNS[0]], color="steelblue", linewidth=.7)
    plt.xlabel("Actual timestamp"); plt.ylabel("AI0 sensor value (unit unverified)")
    plt.title(f"Gap-aware timeline: {frame.source.iloc[0]} (no gap interpolation)")
    save_figure(f"gap_aware_timeline_{frame.source.iloc[0]}.png")
print("Selected structural gap rule:", SEGMENT_GAP_SEC, "seconds; abnormal performance was not consulted.")


# ======== CELL 19 [markdown] ========
# ## 06. Chronological Train / Calibration / Test Split
# 
# Whole normal segments are split before any windows. Split ratios target segment counts (row ratios are reported). Calibration is further divided chronologically into cutoff fitting and policy selection; temporal normal TEST stays untouched until the policy is frozen.
# 

# ======== CELL 20 [code] ========
NORMAL_SPLIT_RATIOS = [0.60, 0.20, 0.20]
CALIBRATION_FIT_FRACTION = 0.70


# ======== CELL 21 [code] ========
assert np.isclose(sum(NORMAL_SPLIT_RATIOS),1)
ordered_ids = normal_segments.sort_values(["segment_start","segment_id"], kind="stable").segment_id.tolist()
if len(ordered_ids) < 12: raise ValueError("Too few acquisition segments for separate TRAIN/CALIBRATION/TEST and policy checks")
boundary1 = max(1,int(len(ordered_ids)*NORMAL_SPLIT_RATIOS[0]))
boundary2 = min(len(ordered_ids)-1,int(len(ordered_ids)*sum(NORMAL_SPLIT_RATIOS[:2])))
split_map = {sid:split for split, ids in zip(["train","calibration","normal_test"],
                  [ordered_ids[:boundary1],ordered_ids[boundary1:boundary2],ordered_ids[boundary2:]]) for sid in ids}
normal["split"] = normal.segment_id.map(split_map)
abnormal["split"] = "abnormal_evaluation"
calibration_ids = ordered_ids[boundary1:boundary2]
cal_boundary = min(len(calibration_ids)-1,max(1,int(len(calibration_ids)*CALIBRATION_FIT_FRACTION)))
calibration_role = {sid:("cutoff_fit" if i < cal_boundary else "policy_selection") for i,sid in enumerate(calibration_ids)}
normal["calibration_role"] = normal.segment_id.map(calibration_role).fillna("not_calibration")
abnormal["calibration_role"] = "not_calibration"
all_rows = pd.concat([normal,abnormal],ignore_index=True)
segment_summary = segment_summary.merge(all_rows.groupby("segment_id").agg(split=("split","first"),
                    calibration_role=("calibration_role","first")).reset_index(), on="segment_id",validate="one_to_one")
assert normal.groupby("segment_id").split.nunique().max() == 1
assert normal.evaluation_label.eq(0).all()
assert set(normal.segment_id).isdisjoint(abnormal.segment_id)
for previous, following in [("train","calibration"),("calibration","normal_test")]:
    assert normal.loc[normal.split.eq(previous),"timestamp"].max() <= normal.loc[normal.split.eq(following),"timestamp"].min()
split_summary = segment_summary.groupby(["source","split","calibration_role"]).agg(
    rows=("segment_n_samples","sum"), segments=("segment_id","size"),
    acquired_duration_sec=("segment_duration_sec","sum"), timestamp_start=("segment_start","min"),
    timestamp_end=("segment_end","max")).reset_index()
display(save_table(split_summary,"split_summary.csv"))
save_table(segment_summary,"segment_summary.csv")
save_table(normal,"processed_normal.csv"); save_table(abnormal,"processed_abnormal.csv")
joblib.dump({"normal":normal,"abnormal":abnormal,"segments":segment_summary}, CACHE_DIR/"processed_data.joblib")
print("Split assigned to raw samples BEFORE window construction. Durations exclude acquisition gaps.")


# ======== CELL 22 [markdown] ========
# ## 07. Normal Operating Regime Analysis
# 
# KMeans is descriptive, fitted on NORMAL TRAIN segment sensor statistics only. Cluster numbers are arbitrary Regime 0/1, not fault labels. Metadata characterize clusters but never enter detectors. Low silhouette or temporal instability is explicitly reported.
# 

# ======== CELL 23 [code] ========
REGIME_K = 2
REGIME_FEATURES = ["current_abs_mean", "AI0_rms", "AI1_rms"]
REGIME_STABILITY_SILHOUETTE_MIN = 0.15  # descriptive check, not proof of physical states


# ======== CELL 24 [code] ========
regime_rows = []
for sid, group in normal.groupby("segment_id",sort=False):
    if not group.valid_measurement.all(): continue
    x = group[SENSOR_COLUMNS].to_numpy()
    regime_rows.append({"segment_id":sid,"current_abs_mean":np.abs(x[:,2]).mean(),
                        "median_current":np.median(x[:,2]), "AI0_rms":np.sqrt(np.mean(x[:,0]**2)),
                        "AI1_rms":np.sqrt(np.mean(x[:,1]**2))})
regime_features = pd.DataFrame(regime_rows).merge(segment_summary,on="segment_id",validate="one_to_one")
regime_train = regime_features.loc[regime_features.split.eq("train")]
assert regime_train.source.eq("normal").all() and regime_train.evaluation_label.eq(0).all()
regime_scaler = RobustScaler().fit(regime_train[REGIME_FEATURES])
regime_model = KMeans(n_clusters=REGIME_K,n_init=20,random_state=RANDOM_STATE).fit(regime_scaler.transform(regime_train[REGIME_FEATURES]))
regime_features["regime"] = regime_model.predict(regime_scaler.transform(regime_features[REGIME_FEATURES]))
regime_silhouette = silhouette_score(regime_scaler.transform(regime_train[REGIME_FEATURES]),regime_model.labels_,sample_size=min(2000,len(regime_train)),random_state=RANDOM_STATE)
REGIMES_SUPPORTED = bool(regime_silhouette >= REGIME_STABILITY_SILHOUETTE_MIN)
regime_summary = regime_features.groupby(["split","regime"]).agg(n_segments=("segment_id","size"),
    sample_count=("segment_n_samples","sum"), median_current=("median_current","median"),
    median_AI0_RMS=("AI0_rms","median"),median_AI1_RMS=("AI1_rms","median"),
    median_duration_sec=("segment_duration_sec","median"),timestamp_start=("segment_start","min"),timestamp_end=("segment_end","max")).reset_index()
regime_summary["proportion_within_split"] = regime_summary.n_segments/regime_summary.groupby("split").n_segments.transform("sum")
display(save_table(regime_summary,"regime_summary.csv"))
regime_map = regime_features.set_index("segment_id").regime.to_dict()
segment_summary["regime"] = segment_summary.segment_id.map(regime_map).fillna(-1).astype(int)
plt.figure()
for regime, group in regime_features.groupby("regime"):
    plt.scatter(group.current_abs_mean,group.AI0_rms,s=12,alpha=.45,label=f"Regime {regime}")
plt.xlabel("Mean absolute current sensor value"); plt.ylabel("AI0 RMS (sensor units)")
plt.title("Normal operating-regime discovery (not fault labels)"); plt.legend(); save_figure("normal_operating_regimes.png")
print(f"TRAIN silhouette={regime_silhouette:.3f}; {'descriptive separation present' if REGIMES_SUPPORTED else 'weak clusters: do not interpret as stable operating regimes'}.")


# ======== CELL 25 [markdown] ========
# ## 08. Physics Feature Engineering
# 
# Nine physical summaries are visible below. Current distortion is residual RMS after least-squares offset+sine+cosine fitting at assumed 0.6Hz, using actual relative timestamps. Adaptive windows are 24/16/8 samples, 50% overlap with an end-anchored last window. <8 has insufficient direct evidence. Current correction is optional and its changes are saved.
# 

# ======== CELL 26 [code] ========
ASSUMED_CURRENT_FREQUENCY_HZ = 0.6
ADAPTIVE_WINDOW_SAMPLES = [24,16,8]
WINDOW_OVERLAP = 0.5
ENABLE_CURRENT_SPIKE_CORRECTION = False
SPIKE_MAD_MULTIPLIER = 8.0
SPIKE_NEIGHBOR_RADIUS = 3
SPIKE_NEIGHBOR_AGREEMENT_MULTIPLIER = 2.0
PHYSICS_FEATURES = ["AI0_rms","AI1_rms","AI0_abs_peak","AI1_abs_peak",
                    "sum_rms","difference_rms","current_center","current_amplitude","current_distortion"]
CURRENT_FEATURES = PHYSICS_FEATURES[-3:]


# ======== CELL 27 [code] ========
def correct_isolated_spikes(frame,apply_correction=None):
    apply_correction=ENABLE_CURRENT_SPIKE_CORRECTION if apply_correction is None else apply_correction
    corrected = frame.copy()
    corrected["AI2_analysis"] = corrected[SENSOR_COLUMNS[2]]
    records = []
    for sid, group in frame.groupby("segment_id",sort=False):
        values = group[SENSOR_COLUMNS[2]].to_numpy(float)
        changes = {}
        scale_floor = max(np.median(np.abs(values-np.median(values)))*1.4826*.05,1e-9)
        for i in range(SPIKE_NEIGHBOR_RADIUS,len(values)-SPIKE_NEIGHBOR_RADIUS):
            neighbors = np.r_[values[i-SPIKE_NEIGHBOR_RADIUS:i],values[i+1:i+SPIKE_NEIGHBOR_RADIUS+1]]
            center = np.median(neighbors)
            scale = max(1.4826*np.median(np.abs(neighbors-center)),scale_floor)
            extreme = abs(values[i]-center) > SPIKE_MAD_MULTIPLIER*scale
            isolated = abs(values[i-1]-values[i+1]) <= SPIKE_NEIGHBOR_AGREEMENT_MULTIPLIER*scale
            if extreme and isolated: changes[i] = .5*(values[i-1]+values[i+1])
        # Adjacent candidates are not defensibly isolated; never correct them.
        for i, replacement in changes.items():
            if i-1 in changes or i+1 in changes: continue
            row_index = group.index[i]
            records.append({"source":group.source.iloc[0],"segment_id":sid,
                            "source_row_id":int(group.source_row_id.iloc[i]),"timestamp":group.timestamp.iloc[i],
                            "original_current":values[i],"replacement_current":replacement,
                            "correction_applied":apply_correction})
            if apply_correction: corrected.loc[row_index,"AI2_analysis"] = replacement
    return corrected,pd.DataFrame(records,columns=["source","segment_id","source_row_id","timestamp","original_current","replacement_current","correction_applied"])

normal_analysis,normal_spikes = correct_isolated_spikes(normal)
abnormal_analysis,abnormal_spikes = correct_isolated_spikes(abnormal)
spike_audit = save_table(pd.concat([normal_spikes,abnormal_spikes]),"current_spike_audit.csv")
print("Isolated spike candidates:",len(spike_audit),"; correction enabled:",ENABLE_CURRENT_SPIKE_CORRECTION)

def sine_features(time_sec, current):
    design = np.column_stack([np.ones(len(time_sec)),np.sin(2*np.pi*ASSUMED_CURRENT_FREQUENCY_HZ*time_sec),
                              np.cos(2*np.pi*ASSUMED_CURRENT_FREQUENCY_HZ*time_sec)])
    coefficients, _, rank, _ = np.linalg.lstsq(design,current,rcond=None)
    fitted = design @ coefficients
    residual = current-fitted
    residual_rms = np.sqrt(np.mean(residual**2))
    variance = np.var(current)
    return {"current_center":coefficients[0],"current_amplitude":np.hypot(*coefficients[1:]),
            "current_distortion":residual_rms,"sine_design_condition":np.linalg.cond(design),
            "sine_r2":1-np.mean(residual**2)/variance if variance > 1e-12 else np.nan,
            "sine_rank":rank}, fitted, residual

def physics_vector(values,time_sec):
    a,b,c = values.T
    features = {"AI0_rms":np.sqrt(np.mean(a*a)),"AI1_rms":np.sqrt(np.mean(b*b)),
                "AI0_abs_peak":np.max(np.abs(a)),"AI1_abs_peak":np.max(np.abs(b)),
                "sum_rms":np.sqrt(np.mean((a+b)**2)),"difference_rms":np.sqrt(np.mean((a-b)**2))}
    current_features,_,_ = sine_features(time_sec,c)
    features.update(current_features)
    return features

def make_windows(frame,fixed_n=None):
    records, waveforms, times = [],{},{}
    # Each split is processed separately; a segment must already have exactly one split.
    for split, split_rows in frame.groupby("split",sort=False):
        for sid, group in split_rows.groupby("segment_id",sort=False):
            assert group.split.nunique()==1 and group.segment_id.nunique()==1
            if not group.valid_measurement.all(): continue
            n = fixed_n if fixed_n else next((n for n in ADAPTIVE_WINDOW_SAMPLES if len(group)>=n),None)
            if n is None or len(group)<n: continue
            step = max(1,int(round(n*(1-WINDOW_OVERLAP))))
            starts = sorted(set(list(range(0,len(group)-n+1,step))+[len(group)-n]))
            for start in starts:
                window = group.iloc[start:start+n]
                dt = window.timestamp.diff().dt.total_seconds().dropna().to_numpy()
                assert (dt>0).all(),"Window crosses duplicate/reversed timestamp"
                time_sec = (window.timestamp-window.timestamp.iloc[0]).dt.total_seconds().to_numpy()
                values = window[[SENSOR_COLUMNS[0],SENSOR_COLUMNS[1],"AI2_analysis"]].to_numpy(float)
                wid = f"{sid}:N{n}:{start}"
                row = {"window_id":wid,"source":window.source.iloc[0],"split":split,
                    "calibration_role":window.calibration_role.iloc[0],"segment_id":sid,
                    "window_n":n,"start_in_segment":start,"position_in_segment":start/max(1,len(group)-1),
                    "start_time":window.timestamp.iloc[0],"available_at":window.timestamp.iloc[-1],
                    "row_ids":window.source_row_id.astype(int).tolist(),"evaluation_label":int(window.evaluation_label.iloc[0]),
                    "segment_n_samples":len(group),"segment_duration_sec":window.segment_duration_sec.iloc[0],
                    "gap_before_sec":window.gap_before_sec.iloc[0],"regime":regime_map.get(sid,-1),
                    "uniform_sampling":bool(np.all(np.abs(dt-NORMAL_MEDIAN_DT)<=UNIFORM_DT_REL_TOL*NORMAL_MEDIAN_DT))}
                row.update(physics_vector(values,time_sec))
                records.append(row);waveforms[wid]=values;times[wid]=time_sec
    return pd.DataFrame(records),waveforms,times

FEATURE_CONFIG = {"inputs":INPUT_HASHES,"gap":SEGMENT_GAP_SEC,"split":NORMAL_SPLIT_RATIOS,
    "calibration_fit":CALIBRATION_FIT_FRACTION,"frequency":ASSUMED_CURRENT_FREQUENCY_HZ,
    "adaptive":ADAPTIVE_WINDOW_SAMPLES,"overlap":WINDOW_OVERLAP,"uniform_tolerance":UNIFORM_DT_REL_TOL,
    "spikes":[ENABLE_CURRENT_SPIKE_CORRECTION,SPIKE_MAD_MULTIPLIER,SPIKE_NEIGHBOR_RADIUS,SPIKE_NEIGHBOR_AGREEMENT_MULTIPLIER],
    "columns":[TIMESTAMP_COLUMN,SENSOR_COLUMNS,LABEL_COLUMN],"regime":[REGIME_K,REGIME_FEATURES],
    "timestamp_utc":TIMESTAMP_PARSE_UTC,"nominal_dt":NOMINAL_DT_SEC}
def generate_physics():
    frames,waves,times = [],{},{}
    for frame in [normal_analysis,abnormal_analysis]:
        features,x,t = make_windows(frame);frames.append(features);waves.update(x);times.update(t)
    return pd.concat(frames,ignore_index=True),waves,times
physics, waveforms, window_times = cached_object("physics_features",FEATURE_CONFIG,generate_physics,
    (generate_physics,make_windows,physics_vector,sine_features,correct_isolated_spikes,construct_segments))
if physics.empty: raise ValueError("No eligible physics windows; inspect segment lengths")
assert physics.groupby("segment_id").split.nunique().max()==1
assert np.isfinite(physics[PHYSICS_FEATURES]).all().all()
save_table(physics,"physics_features.csv")
display(physics.groupby(["source","split","window_n"]).size().rename("windows").reset_index())
feature_summary = physics.groupby("split")[PHYSICS_FEATURES].agg(["median","std","min","max"])
display(feature_summary)
feature_summary_export=feature_summary.copy()
feature_summary_export.columns=[f"{feature}__{statistic}" for feature,statistic in feature_summary_export.columns]
save_table(feature_summary_export.reset_index(),"feature_summary.csv")
print("Adaptive windows created only AFTER split assignment; no metadata or labels enter the feature vector.")


# ======== CELL 28 [markdown] ========
# ## 09. Sine-fit Validation
# 
# 0.6Hz is an assumption, not a proven motor frequency. NORMAL TRAIN fixed N=8/16/24 windows quantify design conditioning and residual-bootstrap uncertainty. A normal-only frequency scan is descriptive and does not alter the assumed frequency automatically. N=8 evidence is flagged when center/amplitude uncertainty is high.
# 

# ======== CELL 29 [code] ========
SINE_STABILITY_N = [8,16,24]
SINE_BOOTSTRAPS = 40
SINE_STABILITY_MAX_WINDOWS = 80
FREQUENCY_SCAN_HZ = [0.3,0.45,0.6,0.75,0.9,1.2]
SINE_MAX_RELATIVE_UNCERTAINTY = 0.5


# ======== CELL 30 [code] ========
rng = np.random.default_rng(RANDOM_STATE)
sine_records, frequency_records = [],[]
train_rows = normal_analysis.loc[normal_analysis.split.eq("train")]
for n in SINE_STABILITY_N:
    fixed, fixed_waves, fixed_times = make_windows(train_rows,fixed_n=n)
    if fixed.empty: continue
    selected = fixed.iloc[np.linspace(0,len(fixed)-1,min(SINE_STABILITY_MAX_WINDOWS,len(fixed)),dtype=int)]
    for _,row in selected.iterrows():
        t = fixed_times[row.window_id]; current = fixed_waves[row.window_id][:,2]
        fitted_features,fitted,residual = sine_features(t,current)
        bootstrap = []
        for _ in range(SINE_BOOTSTRAPS):
            # Conditional iid residual bootstrap: short, dependent residuals limit its inference.
            features,_,_ = sine_features(t,fitted+rng.choice(residual,len(residual),replace=True))
            bootstrap.append([features[c] for c in CURRENT_FEATURES])
        uncertainty = np.std(bootstrap,axis=0)
        scale = max(np.std(current),1e-9)
        sine_records.append({"window_id":row.window_id,"n":n,**fitted_features,
            **{c+"_bootstrap_sd":uncertainty[i] for i,c in enumerate(CURRENT_FEATURES)},
            "center_relative_uncertainty":uncertainty[0]/scale,
            "amplitude_relative_uncertainty":uncertainty[1]/max(fitted_features["current_amplitude"],1e-9)})
        for frequency in FREQUENCY_SCAN_HZ:
            design = np.column_stack([np.ones(n),np.sin(2*np.pi*frequency*t),np.cos(2*np.pi*frequency*t)])
            fit = design @ np.linalg.lstsq(design,current,rcond=None)[0]
            frequency_records.append({"n":n,"frequency_hz":frequency,"relative_residual_rms":np.sqrt(np.mean((current-fit)**2))/scale})
sine_stability = save_table(pd.DataFrame(sine_records),"sine_fit_stability.csv")
sine_summary = sine_stability.groupby("n").median(numeric_only=True).reset_index()
display(save_table(sine_summary,"sine_fit_stability_summary.csv"))
uncertainty_summary=sine_stability.groupby("n")[["center_relative_uncertainty","amplitude_relative_uncertainty"]].quantile(.95).reset_index()
parameter_variability=sine_stability.groupby("n")[CURRENT_FEATURES].std().reset_index()
display(save_table(uncertainty_summary,"sine_fit_uncertainty_q95.csv"))
display(save_table(parameter_variability,"sine_parameter_variability.csv"))
frequency_scan = pd.DataFrame(frequency_records).groupby(["n","frequency_hz"]).relative_residual_rms.median().reset_index()
display(save_table(frequency_scan,"current_frequency_scan.csv"))
current_confidence = {int(row.n): bool(max(row.center_relative_uncertainty,row.amplitude_relative_uncertainty) <= SINE_MAX_RELATIVE_UNCERTAINTY)
                      for row in uncertainty_summary.itertuples()}
physics["current_evidence_low_confidence"] = ~physics.window_n.map(current_confidence).fillna(False)
best_scan_frequency = frequency_scan.loc[frequency_scan.n.eq(24)].sort_values("relative_residual_rms").frequency_hz.iloc[0] if (frequency_scan.n==24).any() else np.nan
SINE_ASSUMPTION_SUPPORTED = bool(np.isclose(best_scan_frequency,ASSUMED_CURRENT_FREQUENCY_HZ))
print("Normal-only best scanned N=24 frequency:",best_scan_frequency,"Hz; 0.6Hz reproduced:",SINE_ASSUMPTION_SUPPORTED)
print("Confidence by N:",current_confidence,". Bootstrap uncertainty is conditional, not calibrated physical uncertainty.")
example = physics.loc[physics.split.eq("train")].iloc[0]
t,current = window_times[example.window_id],waveforms[example.window_id][:,2]
_,fitted,_ = sine_features(t,current)
plt.figure();plt.plot(t,current,"o-",label="Recorded current");plt.plot(t,fitted,label="Assumed 0.6 Hz fit")
plt.xlabel("Actual relative time (seconds)");plt.ylabel("Current sensor value (unit unverified)")
plt.title("Normal current: sine-fit example");plt.legend();save_figure("current_sine_fit_example.png")


# ======== CELL 31 [markdown] ========
# ## 10. Robust Joint Mahalanobis
# 
# RobustScaler and MCD fit NORMAL TRAIN only. Report robust support and covariance condition. Explicit eigenvalue flooring stabilizes the precision matrix if required; it is disclosed, not silently hidden. All nine features are retained; no direct vibration correlation feature is used.
# 

# ======== CELL 32 [code] ========
MCD_SUPPORT_FRACTION = 0.75
COVARIANCE_EIGEN_FLOOR_RELATIVE = 1e-8
MIN_TRAIN_WINDOWS = 30


# ======== CELL 33 [code] ========
def fit_mcd_reference(features,columns):
    if len(features)<max(MIN_TRAIN_WINDOWS,3*len(columns)):
        raise ValueError(f"Insufficient normal TRAIN support: {len(features)} windows / {len(columns)} features")
    assert features.source.eq("normal").all() and features.split.eq("train").all()
    assert features.evaluation_label.eq(0).all()
    scaler = RobustScaler().fit(features[columns])
    scaled = scaler.transform(features[columns])
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        model = MinCovDet(support_fraction=MCD_SUPPORT_FRACTION,random_state=RANDOM_STATE).fit(scaled)
    eigenvalues,eigenvectors = np.linalg.eigh(model.covariance_)
    floor = max(float(eigenvalues.max())*COVARIANCE_EIGEN_FLOOR_RELATIVE,1e-12)
    regularized = np.maximum(eigenvalues,floor)
    precision = (eigenvectors / regularized) @ eigenvectors.T
    diagnostics = {"n_train":len(features),"n_features":len(columns),"support_fraction":MCD_SUPPORT_FRACTION,
        "raw_support":int(model.raw_support_.sum()),"effective_support":int(model.support_.sum()),
        "condition_before":float(np.linalg.cond(model.covariance_)),"condition_after":float(regularized.max()/regularized.min()),
        "eigenvalues_floored":int((eigenvalues<floor).sum()),"warnings":[str(w.message) for w in recorded_warnings]}
    return {"scaler":scaler,"model":model,"precision":precision,"columns":columns,"diagnostics":diagnostics,
            "fit_source":"normal", "fit_split":"train"}

def mcd_score(reference,features):
    delta = reference["scaler"].transform(features[reference["columns"]])-reference["model"].location_
    # Squared distance, consistent with sklearn .mahalanobis; not a probability.
    return np.maximum(np.einsum("ij,jk,ik->i",delta,reference["precision"],delta),0)

physics_train = physics.loc[physics.split.eq("train")].copy()
MODEL_CONFIG = {**FEATURE_CONFIG,"support":MCD_SUPPORT_FRACTION,"floor":COVARIANCE_EIGEN_FLOOR_RELATIVE,"min_train":MIN_TRAIN_WINDOWS,
                "physics_columns":PHYSICS_FEATURES,"current_columns":CURRENT_FEATURES}
joint_reference = cached_object("joint_mcd",MODEL_CONFIG,lambda:fit_mcd_reference(physics_train,PHYSICS_FEATURES),
                               (fit_mcd_reference,mcd_score,physics_vector,sine_features))
display(physics_train[PHYSICS_FEATURES].agg(["median","std","min","max"]))
display(pd.DataFrame([joint_reference["diagnostics"]]))
print("Joint reference trained exclusively on",len(physics_train),"normal windows; scores are squared robust distances.")


# ======== CELL 34 [markdown] ========
# ## 11. Current-only Mahalanobis
# 
# Three current features form a separate normal reference. Joint and current evidence remain separate; there is no automatic score averaging.
# 

# ======== CELL 35 [code] ========
CURRENT_MODEL_COLUMNS = CURRENT_FEATURES.copy()


# ======== CELL 36 [code] ========
current_reference = cached_object("current_mcd",{**MODEL_CONFIG,"columns":CURRENT_MODEL_COLUMNS},
    lambda:fit_mcd_reference(physics_train,CURRENT_MODEL_COLUMNS),(fit_mcd_reference,mcd_score,physics_vector,sine_features))
display(pd.DataFrame([current_reference["diagnostics"]]))
covariance_diagnostics = save_table(pd.DataFrame([{"model":"joint",**joint_reference["diagnostics"]},
                                                {"model":"current",**current_reference["diagnostics"]}]),"covariance_diagnostics.csv")
print("Current reference is diagnostic evidence. Short-window fit uncertainty is retained as a confidence flag.")


# ======== CELL 37 [markdown] ========
# ## 12. Optional Normalized Waveform Detector
# 
# Normalize each sensor within each window to remove magnitude/offset, then resample normalized values on relative time to a common vector. Shrinkage covariance fits NORMAL TRAIN only. Irregularly sampled windows are ineligible; no gap interpolation is performed. Both normalizations are a fixed, small comparison.
# 

# ======== CELL 38 [code] ========
ENABLE_WAVEFORM_BRANCH = True
WAVEFORM_NORMALIZATIONS = ["mean_std","median_mad"]
SHAPE_RESAMPLE_POINTS = 24
NORMALIZATION_SCALE_FLOOR = 1e-8


# ======== CELL 39 [code] ========
def shape_vector(values,time_sec,method):
    center = values.mean(axis=0) if method=="mean_std" else np.median(values,axis=0)
    scale = values.std(axis=0) if method=="mean_std" else 1.4826*np.median(np.abs(values-center),axis=0)
    normalized = (values-center)/np.maximum(scale,NORMALIZATION_SCALE_FLOOR)
    grid = np.linspace(time_sec[0],time_sec[-1],SHAPE_RESAMPLE_POINTS)
    return np.concatenate([np.interp(grid,time_sec,normalized[:,channel]) for channel in range(3)])

waveform_references = {}
if ENABLE_WAVEFORM_BRANCH:
    eligible_train = physics_train.loc[physics_train.uniform_sampling]
    for method in WAVEFORM_NORMALIZATIONS:
        matrix = np.vstack([shape_vector(waveforms[wid],window_times[wid],method) for wid in eligible_train.window_id])
        waveform_references[method] = LedoitWolf().fit(matrix)
        print(method,"normal TRAIN support:",len(matrix),"; shrinkage:",waveform_references[method].shrinkage_)
    joblib.dump(waveform_references,CACHE_DIR/"waveform_references.joblib")
else: print("Waveform branch disabled explicitly.")
joblib.dump(waveform_references,CACHE_DIR/"waveform_references.joblib")


# ======== CELL 40 [markdown] ========
# ## 13. Optional Low-frequency Spectral Detector
# 
# Current: mean removal → Hann taper → rFFT → remove DC → normalize power. Only observed 0–Nyquist low-frequency shape is interpreted; anti-aliasing and instantaneous-current status are unverified. N=20 and 24 are separate evidence experiments. Choose k using later NORMAL CALIBRATION segment false-warning stability only.
# 

# ======== CELL 41 [code] ========
ENABLE_SPECTRAL_BRANCH = True
SPECTRAL_WINDOW_SAMPLES = [20,24]
SPECTRAL_NEIGHBORS = [3,5,7]
SPECTRAL_TAPER = "hann"
SPECTRAL_ZERO_POWER_FLOOR = 1e-12


# ======== CELL 42 [code] ========
def spectrum_vector(values,taper="hann"):
    current = values[:,2]-values[:,2].mean()
    weights = np.hanning(len(current)) if taper=="hann" else np.ones(len(current))
    power = np.abs(np.fft.rfft(current*weights))**2
    power = power[1:]  # explicitly exclude DC after taper
    if power.sum() <= SPECTRAL_ZERO_POWER_FLOOR: return None
    return power/power.sum()

spectral_references, spectral_frames, spectral_waveforms = {},{},{}
spectral_selection = []
if ENABLE_SPECTRAL_BRANCH:
    for n in SPECTRAL_WINDOW_SAMPLES:
        frame,waves,times = make_windows(pd.concat([normal_analysis,abnormal_analysis]),fixed_n=n)
        if frame.empty: continue
        eligible_ids = [wid for wid in frame.loc[frame.uniform_sampling,"window_id"] if spectrum_vector(waves[wid],SPECTRAL_TAPER) is not None]
        frame = frame.loc[frame.window_id.isin(eligible_ids)].copy()
        if frame.empty: continue
        matrix = np.vstack([spectrum_vector(waves[wid],SPECTRAL_TAPER) for wid in frame.window_id])
        train_mask = frame.split.eq("train").to_numpy()
        cutoff_mask = (frame.split.eq("calibration") & frame.calibration_role.eq("cutoff_fit")).to_numpy()
        selection_mask = (frame.split.eq("calibration") & frame.calibration_role.eq("policy_selection")).to_numpy()
        if train_mask.sum()<max(SPECTRAL_NEIGHBORS) or cutoff_mask.sum()<5 or selection_mask.sum()<2:
            print("Spectral N",n,"has insufficient normal support; skipped.");continue
        for k in SPECTRAL_NEIGHBORS:
            knn = NearestNeighbors(n_neighbors=k).fit(matrix[train_mask])
            distances = knn.kneighbors(matrix)[0].mean(axis=1)
            threshold = np.quantile(distances[cutoff_mask],.99)
            segment_rates = frame.loc[selection_mask,["segment_id","start_time"]].assign(flag=distances[selection_mask]>threshold)
            segment_flags = segment_rates.groupby("segment_id").flag.max()
            blocks = np.array_split(segment_flags.to_numpy(),3)
            stability = np.std([block.mean() for block in blocks if len(block)])
            spectral_selection.append({"n":n,"k":k,"selection_segment_false_rate":segment_flags.mean(),
                "normal_block_rate_sd":stability,"cutoff_fit_windows":int(cutoff_mask.sum())})
        rank = pd.DataFrame(spectral_selection).query("n==@n").sort_values(["selection_segment_false_rate","normal_block_rate_sd","k"])
        k = int(rank.iloc[0].k)
        spectral_references[n] = {"model":NearestNeighbors(n_neighbors=k).fit(matrix[train_mask]),"k":k,"n":n,"taper":SPECTRAL_TAPER}
        frame["spectrum_score"] = spectral_references[n]["model"].kneighbors(matrix)[0].mean(axis=1)
        spectral_frames[n],spectral_waveforms[n] = frame,waves
    display(save_table(pd.DataFrame(spectral_selection),"spectral_normal_selection.csv"))
    print("Spectral k selection used only normal calibration; abnormal scores did not enter selection.")
else: print("Spectral branch disabled explicitly.")
save_table(pd.DataFrame(spectral_selection,columns=["n","k","selection_segment_false_rate","normal_block_rate_sd","cutoff_fit_windows"]),
           "spectral_normal_selection.csv")


# ======== CELL 43 [markdown] ========
# ## 14. Threshold Calibration
# 
# Cutoffs are frozen from the early chronological NORMAL CALIBRATION subset. The later calibration subset chooses alarm policy; test and abnormal rows do not affect any cutoff. Quantile tails have limited support, and overlapping windows reduce effective sample size.
# 

# ======== CELL 44 [code] ========
CALIBRATION_QUANTILES = [0.90,0.95,0.975,0.99]
ATTENTION_QUANTILE = 0.975
WARNING_QUANTILE = 0.99
MIN_CUTOFF_WINDOWS = 20


# ======== CELL 45 [code] ========
def score_components(feature_frame,waves,times):
    result = feature_frame.copy()
    result["joint_score"] = mcd_score(joint_reference,feature_frame)
    result["current_score"] = mcd_score(current_reference,feature_frame)
    for method, model in waveform_references.items():
        result["shape_"+method+"_score"] = np.nan
        eligible = feature_frame.uniform_sampling
        if eligible.any():
            matrix = np.vstack([shape_vector(waves[wid],times[wid],method) for wid in feature_frame.loc[eligible,"window_id"]])
            result.loc[eligible,"shape_"+method+"_score"] = model.mahalanobis(matrix)
    result["spectrum_score"] = np.nan
    for n, reference in spectral_references.items():
        eligible = feature_frame.uniform_sampling & feature_frame.window_n.eq(n)
        ids = [wid for wid in feature_frame.loc[eligible,"window_id"] if spectrum_vector(waves[wid],reference["taper"]) is not None]
        if ids:
            matrix = np.vstack([spectrum_vector(waves[wid],reference["taper"]) for wid in ids])
            scores_n = reference["model"].kneighbors(matrix)[0].mean(axis=1)
            result.loc[result.window_id.isin(ids),"spectrum_score"] = scores_n
    return result

scores = score_components(physics,waveforms,window_times)
score_columns = [c for c in scores.columns if c.endswith("_score") and scores[c].notna().any()]
cutoff_rows = scores.loc[scores.split.eq("calibration") & scores.calibration_role.eq("cutoff_fit")]
assert cutoff_rows.source.eq("normal").all() and cutoff_rows.evaluation_label.eq(0).all()
if len(cutoff_rows)<MIN_CUTOFF_WINDOWS: raise ValueError("Insufficient normal calibration windows; no defensible cutoffs")
thresholds,threshold_records = {},[]
for column in score_columns:
    values = cutoff_rows[column].dropna()
    if len(values)<MIN_CUTOFF_WINDOWS:
        print("Too few cutoff values; branch excluded:",column);continue
    thresholds[column] = {str(q):float(np.quantile(values,q)) for q in CALIBRATION_QUANTILES}
    for q,value in thresholds[column].items():
        if value<=0: raise ValueError(f"Nonpositive cutoff for {column}; inspect degenerate reference")
        threshold_records.append({"detector":column,"quantile":float(q),"cutoff":value,
                                  "source":"normal","split":"calibration","role":"cutoff_fit",
                                  "n_windows":len(values),"nominal_tail_windows":len(values)*(1-float(q))})
threshold_table = save_table(pd.DataFrame(threshold_records),"alarm_thresholds.csv")
display(threshold_table)
write_json(MODEL_DIR/"alarm_thresholds.json",thresholds)
save_table(scores,"scores.csv");joblib.dump(scores,CACHE_DIR/"scores.joblib")
print("All cutoffs frozen before abnormal evaluation. q99 tails need substantially more independent normal episodes.")


# ======== CELL 46 [markdown] ========
# ## 15. Segment Aggregation
# 
# Overlapping windows are not independent events. Compare maximum, top-two mean and fraction exceeding the fixed window cutoff. Include unscored segments explicitly; missing evidence is not a confident NORMAL decision.
# 

# ======== CELL 47 [code] ========
SEGMENT_AGGREGATIONS = ["max","top2_mean","fraction_above"]
SEGMENT_FRACTION_TRIGGER = 0.5


# ======== CELL 48 [code] ========
def aggregate_scores(window_scores,summary):
    records = []
    for sid,group in window_scores.groupby("segment_id",sort=False):
        record = {"segment_id":sid,"n_windows":len(group)}
        for column in thresholds:
            values = group[column].dropna().to_numpy()
            cutoff = thresholds[column][str(WARNING_QUANTILE)]
            record[column+"_max"] = np.max(values) if len(values) else np.nan
            record[column+"_top2_mean"] = np.mean(np.sort(values)[-2:]) if len(values) else np.nan
            record[column+"_fraction_above"] = np.mean(values>cutoff) if len(values) else np.nan
        records.append(record)
    aggregated = summary.merge(pd.DataFrame(records),on="segment_id",how="left",validate="one_to_one")
    aggregated["n_windows"] = aggregated.n_windows.fillna(0).astype(int)
    aggregated["direct_evidence"] = aggregated.n_windows.gt(0)
    return aggregated

segment_scores = aggregate_scores(scores,segment_summary)
save_table(segment_scores,"segment_score_aggregations.csv")
display(segment_scores.groupby(["split","direct_evidence"]).size().rename("segments").reset_index())
print("Max is sensitive to window count; top-2 stabilizes one-window extremes; fraction requires repeated evidence.")


# ======== CELL 49 [markdown] ========
# ## 16. Alarm State Machine
# 
# Three evidence states are NORMAL/ATTENTION/WARNING; insufficient evidence is a separate QC flag. Moderate evidence yields ATTENTION, extreme joint evidence plus a complementary detector can yield instant WARNING, and persistence compares none/2-of-3/EWMA. State resets at every acquisition gap. Selection uses only later calibration segments and penalizes lost primary evidence in TRAIN; abnormal data is not inspected.
# 

# ======== CELL 50 [code] ========
ALARM_POLICIES = ["none","two_of_three","ewma"]
EWMA_ALPHA = 0.4
POLICY_MAX_CALIBRATION_WARNING_RATE = 0.10
POLICY_EVIDENCE_RETENTION_MIN = 0.20  # prevent selecting an effectively disabled confirmation rule
POLICY_MIN_DIRECT_COVERAGE = 0.80
PRIMARY_DEFAULT = "joint_score"


# ======== CELL 51 [code] ========
candidate_definitions = {"A_physics":("joint_score",None),"B_current":("current_score",None)}
for method in WAVEFORM_NORMALIZATIONS:
    column = "shape_"+method+"_score"
    if column in thresholds: candidate_definitions["C_shape_"+method]=(column,None)
if "spectrum_score" in thresholds: candidate_definitions["D_spectrum"]=("spectrum_score",None)
candidate_definitions["E_physics_current"]=("joint_score","current_score")
if "shape_mean_std_score" in thresholds:
    candidate_definitions["F_physics_shape"]=("joint_score","shape_mean_std_score")
if "spectrum_score" in thresholds:
    candidate_definitions["F_physics_spectrum"]=("joint_score","spectrum_score")

def candidate_margin(frame,candidate):
    primary,confirmation = candidate_definitions[candidate]
    primary_margin = frame[primary]/thresholds[primary][str(WARNING_QUANTILE)]
    if confirmation:
        companion = frame[confirmation]/thresholds[confirmation][str(WARNING_QUANTILE)]
        return np.minimum(primary_margin,companion)  # logical confirmation, not probability averaging
    return primary_margin

def smooth_margin(frame,margin):
    return pd.Series(margin,index=frame.index).groupby(frame.segment_id,sort=False).transform(
        lambda group:group.ewm(alpha=EWMA_ALPHA,adjust=False).mean())

ewma_thresholds = {}
for candidate in candidate_definitions:
    margin = candidate_margin(scores,candidate)
    smoothed = smooth_margin(scores,margin)
    values = smoothed.loc[cutoff_rows.index].dropna()
    if len(values): ewma_thresholds[candidate]=float(np.quantile(values,WARNING_QUANTILE))

def alarm_windows(frame,candidate,policy):
    primary,confirmation = candidate_definitions[candidate]
    result = frame.sort_values(["segment_id","available_at","window_id"],kind="stable").copy()
    moderate = result[primary] > thresholds[primary][str(ATTENTION_QUANTILE)]
    if confirmation:
        moderate = moderate | (result[confirmation] > thresholds[confirmation][str(ATTENTION_QUANTILE)])
    extreme = result[primary] > thresholds[primary][str(WARNING_QUANTILE)]
    if confirmation:
        instant = extreme & (result[confirmation] > thresholds[confirmation][str(ATTENTION_QUANTILE)])
        # Require repeated primary deviation AND at least moderate companion evidence.
        persistent_evidence = moderate & (result[primary] > thresholds[primary][str(ATTENTION_QUANTILE)]) & result[confirmation].gt(thresholds[confirmation][str(ATTENTION_QUANTILE)])
    else:
        instant = pd.Series(False,index=result.index)
        persistent_evidence = moderate
    margin = candidate_margin(result,candidate)
    if policy=="none": warning = instant | (margin>1)
    elif policy=="two_of_three":
        counts = persistent_evidence.astype(int).groupby(result.segment_id,sort=False).transform(lambda x:x.rolling(3,min_periods=2).sum())
        warning = instant | counts.ge(2)
    elif policy=="ewma": warning = instant | smooth_margin(result,margin).gt(ewma_thresholds[candidate])
    else: raise ValueError(policy)
    result["candidate"],result["policy"] = candidate,policy
    result["warning"] = warning.fillna(False)
    result["attention"] = moderate.fillna(False) | result.warning  # inclusive review burden
    result["candidate_score"] = margin
    result["state"] = np.where(result.warning,"WARNING",np.where(result.attention,"ATTENTION","NORMAL"))
    result["candidate_evidence_available"] = result[primary].notna()
    if confirmation: result["candidate_evidence_available"] &= result[confirmation].notna()
    return result

def segment_decisions(windows,summary):
    grouped = windows.groupby("segment_id").agg(warning=("warning","max"),attention=("attention","max"),
        candidate_score=("candidate_score","max"),n_windows=("window_id","size"),
        candidate_evidence_available=("candidate_evidence_available","any"))
    first_warning = windows.loc[windows.warning].groupby("segment_id").available_at.min()
    grouped["first_warning_at"] = first_warning
    result = summary.merge(grouped.reset_index(),on="segment_id",how="left",validate="one_to_one")
    for column in ["warning","attention","candidate_evidence_available"]: result[column]=result[column].fillna(False).astype(bool)
    result["n_windows"]=result.n_windows.fillna(0).astype(int)
    result["direct_evidence"] = result.candidate_evidence_available
    result["evidence_status"] = np.where(result.direct_evidence,"DIRECT","INSUFFICIENT")
    result["state"] = np.where(result.warning,"WARNING",np.where(result.attention,"ATTENTION","NORMAL"))
    result["first_warning_delay_sec"] = (result.first_warning_at-result.segment_start).dt.total_seconds()
    return result

selection_summary = segment_summary.loc[segment_summary.calibration_role.eq("policy_selection")]
selection_scores = scores.loc[scores.calibration_role.eq("policy_selection")]
policy_records = []
for candidate in candidate_definitions:
    primary,_ = candidate_definitions[candidate]
    for policy in ALARM_POLICIES:
        windows = alarm_windows(selection_scores,candidate,policy)
        segments = segment_decisions(windows,selection_summary)
        observed = segments.loc[segments.direct_evidence]
        by_regime = observed.groupby("regime").warning.mean()
        train_evidence = alarm_windows(scores.loc[scores.split.eq("train")],candidate,policy)
        base_train = scores.loc[scores.split.eq("train"),primary]>thresholds[primary][str(WARNING_QUANTILE)]
        retention = (train_evidence.warning & base_train.reindex(train_evidence.index)).sum()/max(1,base_train.sum())
        blocks = np.array_split(observed.sort_values("segment_start").warning.to_numpy(),3)
        policy_records.append({"candidate":candidate,"policy":policy,"normal_selection_segments":len(observed),
            "warning_rate":observed.warning.mean() if len(observed) else np.nan,
            "attention_rate":observed.attention.mean() if len(observed) else np.nan,
            "worst_regime_warning_rate":by_regime.max() if len(by_regime) else np.nan,
            "normal_block_rate_sd":np.std([b.mean() for b in blocks if len(b)]),
            "direct_coverage":segments.direct_evidence.mean(),"train_primary_tail_retention":retention,
            "complexity":(0 if candidate=="A_physics" else 1)+(1 if candidate.startswith(("E_","F_")) else 0)+(policy!="none")})
policy_selection = pd.DataFrame(policy_records)
eligible = policy_selection.loc[(policy_selection.direct_coverage>=POLICY_MIN_DIRECT_COVERAGE) &
    (policy_selection.train_primary_tail_retention>=POLICY_EVIDENCE_RETENTION_MIN)].copy()
if eligible.empty: eligible=policy_selection.loc[policy_selection.candidate.eq("A_physics")].copy()
# Honor the primary physical representation; secondary detectors are diagnostic comparators.
eligible = eligible.loc[eligible.candidate.str.startswith(("A_","E_","F_"))]
ranked = eligible.sort_values(["warning_rate","worst_regime_warning_rate","normal_block_rate_sd","complexity"],kind="stable")
FINAL_CANDIDATE,FINAL_POLICY = ranked.iloc[0][["candidate","policy"]].tolist()
FINAL_SELECTION_LOCK = {"candidate":FINAL_CANDIDATE,"policy":FINAL_POLICY,"thresholds":thresholds,
                        "selected_from":"normal calibration policy_selection segments","abnormal_used":False,
                        "normal_test_used":False,"timestamp":str(pd.Timestamp.now(tz="UTC"))}
write_json(MODEL_DIR/"selection_lock.json",FINAL_SELECTION_LOCK)
display(save_table(policy_selection,"alarm_policy_selection.csv"))
print("Frozen BEFORE test/abnormal results:",FINAL_CANDIDATE,FINAL_POLICY)
print("Selection checks false-warning burden and regime stability; TRAIN tail retention is only a non-degeneracy check, not fault sensitivity.")
if ranked.iloc[0].warning_rate>POLICY_MAX_CALIBRATION_WARNING_RATE:
    print("WARNING: normal selection burden exceeds configured review budget; no deployment readiness claim.")


# ======== CELL 52 [markdown] ========
# ## 17. Temporal Normal Evaluation
# 
# The policy is already locked. Evaluate chronologically later normal segments and compare fixed alternatives descriptively. These alternatives do not retune the final policy. Episodes merge consecutive WARNING windows within a segment, never across acquisition gaps.
# 

# ======== CELL 53 [code] ========
TEMPORAL_NORMAL_BLOCKS = 4
EVALUATE_ALL_FIXED_CANDIDATES = True


# ======== CELL 54 [code] ========
def alarm_episode_table(windows):
    records = []
    for sid,group in windows.groupby("segment_id",sort=False):
        group = group.sort_values("available_at")
        active = group.warning.to_numpy(bool)
        starts = np.flatnonzero(active & ~np.r_[False,active[:-1]])
        ends = np.flatnonzero(active & ~np.r_[active[1:],False])
        for number,(start,end) in enumerate(zip(starts,ends)):
            chunk = group.iloc[start:end+1]
            records.append({"segment_id":sid,"episode_id":f"{sid}:alarm{number}",
                "source":chunk.source.iloc[0],"split":chunk.split.iloc[0],
                "first_available_at":chunk.available_at.iloc[0],"last_available_at":chunk.available_at.iloc[-1],
                "alarm_duration_sec":(chunk.available_at.iloc[-1]-chunk.available_at.iloc[0]).total_seconds(),
                "evidence_interval_duration_sec":(chunk.available_at.iloc[-1]-chunk.start_time.iloc[0]).total_seconds(),
                "warning_windows":len(chunk),"evaluation_label":int(chunk.evaluation_label.iloc[0])})
    return pd.DataFrame(records,columns=["segment_id","episode_id","source","split","first_available_at",
        "last_available_at","alarm_duration_sec","evidence_interval_duration_sec","warning_windows","evaluation_label"])

def normal_metrics(windows,segments,candidate,policy):
    eligible = segments.loc[segments.direct_evidence]
    episodes = alarm_episode_table(windows)
    return {"candidate":candidate,"policy":policy,"split":segments.split.iloc[0],
        "n_segments_all":len(segments),"n_segments_direct":len(eligible),
        "normal_false_warning_segment_rate":eligible.warning.mean() if len(eligible) else np.nan,
        "normal_attention_segment_rate":eligible.attention.mean() if len(eligible) else np.nan,
        "normal_false_warning_segment_rate_all":segments.warning.mean(),
        "warning_windows":int(windows.warning.sum()),"alarm_episodes":len(episodes),
        "maximum_consecutive_false_positive_run":int(episodes.warning_windows.max()) if len(episodes) else 0,
        "episode_duration_sec_total":episodes.alarm_duration_sec.sum() if len(episodes) else 0,
        "direct_segment_coverage":segments.direct_evidence.mean()}

normal_test_summary = segment_summary.loc[segment_summary.split.eq("normal_test")]
normal_test_scores = scores.loc[scores.split.eq("normal_test")]
normal_results = []
for candidate in candidate_definitions:
    for policy in ALARM_POLICIES:
        windows = alarm_windows(normal_test_scores,candidate,policy)
        decisions = segment_decisions(windows,normal_test_summary)
        normal_results.append(normal_metrics(windows,decisions,candidate,policy))
normal_evaluation = save_table(pd.DataFrame(normal_results),"temporal_normal_evaluation.csv")
normal_final_windows = alarm_windows(normal_test_scores,FINAL_CANDIDATE,FINAL_POLICY)
normal_final_segments = segment_decisions(normal_final_windows,normal_test_summary)
normal_episodes = alarm_episode_table(normal_final_windows)
final_normal_result = normal_evaluation.loc[normal_evaluation.candidate.eq(FINAL_CANDIDATE)&normal_evaluation.policy.eq(FINAL_POLICY)].iloc[0]
display(final_normal_result.to_frame("value"))
block_table = normal_test_scores.copy().sort_values("available_at")
block_table["temporal_block"] = np.minimum(TEMPORAL_NORMAL_BLOCKS-1,np.arange(len(block_table))*TEMPORAL_NORMAL_BLOCKS//max(1,len(block_table)))
save_table(block_table.groupby("temporal_block")[list(thresholds)].agg(["count","median","max"]).reset_index(),"normal_temporal_score_blocks.csv")
print(f"Locked policy: later-normal false-warning segment rate={final_normal_result.normal_false_warning_segment_rate:.3f}, "
      f"Attention rate={final_normal_result.normal_attention_segment_rate:.3f}, episodes={len(normal_episodes)}.")
print("Normal TEST supports a within-recording temporal check, not independent-site generalization.")


# ======== CELL 55 [markdown] ========
# ## 18. Recorded Abnormal Evaluation
# 
# First evaluation of the abnormal-state recording under locked settings. It is one recording, not a series of independent mechanical failures. Report recall among directly scored segments AND all supplied segments. PR-AUC and precision use the artificial mixture of later-normal and abnormal records, so deployment prevalence is not represented.
# 

# ======== CELL 56 [code] ========
ABNORMAL_EVALUATION_NAME = "provided recorded abnormal state"


# ======== CELL 57 [code] ========
abnormal_final_windows = alarm_windows(scores.loc[scores.split.eq("abnormal_evaluation")],FINAL_CANDIDATE,FINAL_POLICY)
abnormal_final_segments = segment_decisions(abnormal_final_windows,segment_summary.loc[segment_summary.source.eq("abnormal")])
final_windows = pd.concat([normal_final_windows,abnormal_final_windows],ignore_index=True)
final_segments = pd.concat([normal_final_segments,abnormal_final_segments],ignore_index=True)
final_episodes = alarm_episode_table(final_windows)

def mixture_metrics(frame,level,candidate,policy):
    eligible = frame.loc[frame.candidate_evidence_available].copy()
    y = eligible.evaluation_label.to_numpy(int); predicted = eligible.warning.to_numpy(int)
    precision,recall,f1,_ = precision_recall_fscore_support(y,predicted,average="binary",zero_division=0)
    two_classes = len(np.unique(y))==2
    return {"candidate":candidate,"policy":policy,"level":level,"n_evidence":len(eligible),
        "n_total":len(frame),"precision_mixture":precision,"recall_direct":recall,"f1_mixture":f1,
        "pr_auc_mixture":average_precision_score(y,eligible.candidate_score) if two_classes else np.nan,
        "true_positive":int(((y==1)&(predicted==1)).sum()),"false_negative":int(((y==1)&(predicted==0)).sum()),
        "false_positive":int(((y==0)&(predicted==1)).sum()),"true_negative":int(((y==0)&(predicted==0)).sum())}

recorded_metrics = pd.DataFrame([mixture_metrics(final_windows,"window",FINAL_CANDIDATE,FINAL_POLICY),
    mixture_metrics(final_segments.rename(columns={"direct_evidence":"direct_evidence_segment"}),"segment",FINAL_CANDIDATE,FINAL_POLICY)])
direct_abnormal = abnormal_final_segments.loc[abnormal_final_segments.direct_evidence]
abnormal_segment_recall_direct = direct_abnormal.warning.mean() if len(direct_abnormal) else np.nan
abnormal_segment_recall_all = abnormal_final_segments.warning.mean()
save_table(recorded_metrics,"recorded_evaluation.csv")
save_table(final_segments,"segment_predictions.csv")
save_table(final_windows,"window_predictions.csv")
save_table(final_episodes,"alarm_episodes.csv")
display(recorded_metrics)
print(f"Recorded abnormal segments: {int(abnormal_final_segments.warning.sum())}/{len(direct_abnormal)} directly eligible detected; "
      f"{len(abnormal_final_segments)-len(direct_abnormal)} segments lack direct candidate evidence.")
print(f"Recall among directly eligible segments={abnormal_segment_recall_direct:.3f}; all-recorded-segment recall={abnormal_segment_recall_all:.3f}.")
print("Alarm onset uses window completion timestamp. This is retrospective detection delay, not pre-failure lead time.")


# ======== CELL 58 [markdown] ========
# ## 19. Operating-Regime Robustness
# 
# Evaluate every frozen candidate within normal test regimes. A high-current cluster is never labeled a fault. Also remove each physics feature in a normal-only reference to expose regime-dependent burden; this diagnostic does not change the locked model after viewing abnormal results.
# 

# ======== CELL 59 [code] ========
REGIME_FEATURE_REMOVAL_DIAGNOSTIC = True
REGIME_FALSE_WARNING_DISPARITY_LIMIT = 0.10


# ======== CELL 60 [code] ========
regime_evaluation_records = []
for candidate in candidate_definitions:
    windows = alarm_windows(normal_test_scores,candidate,FINAL_POLICY)
    segments = segment_decisions(windows,normal_test_summary)
    for regime,group in segments.groupby("regime"):
        sensor_metadata = regime_features.loc[regime_features.segment_id.isin(group.segment_id)]
        eligible = group.loc[group.direct_evidence]
        regime_evaluation_records.append({"candidate":candidate,"regime":regime,"n_segments":len(group),
            "n_direct_segments":len(eligible),"median_current":sensor_metadata.median_current.median(),
            "median_vibration":sensor_metadata.AI0_rms.median(),"mean_anomaly_score":eligible.candidate_score.mean(),
            "attention_rate":eligible.attention.mean() if len(eligible) else np.nan,
            "warning_rate":eligible.warning.mean() if len(eligible) else np.nan})
regime_evaluation = save_table(pd.DataFrame(regime_evaluation_records),"regime_false_alarm_rates.csv")
display(regime_evaluation)
feature_removal_records = []
if REGIME_FEATURE_REMOVAL_DIAGNOSTIC:
    for removed in PHYSICS_FEATURES:
        columns = [column for column in PHYSICS_FEATURES if column!=removed]
        reference = fit_mcd_reference(physics_train,columns)
        cutoff = np.quantile(mcd_score(reference,cutoff_rows),WARNING_QUANTILE)
        diagnostic = normal_test_scores[["segment_id","regime"]].copy()
        diagnostic["flag"] = mcd_score(reference,normal_test_scores)>cutoff
        for regime,group in diagnostic.groupby("regime"):
            feature_removal_records.append({"removed_feature":removed,"regime":regime,
                                           "normal_segment_exceedance_rate":group.groupby("segment_id").flag.max().mean()})
display(save_table(pd.DataFrame(feature_removal_records),"regime_feature_removal.csv"))
final_regime_rates = regime_evaluation.loc[regime_evaluation.candidate.eq(FINAL_CANDIDATE)]
regime_disparity = final_regime_rates.warning_rate.max()-final_regime_rates.warning_rate.min()
print(f"Locked candidate regime false-warning disparity={regime_disparity:.3f}; "
      f"{'within configured descriptive tolerance' if regime_disparity<=REGIME_FALSE_WARNING_DISPARITY_LIMIT else 'regime dependence remains'}.")
print("Cluster separation does not establish stable physical operating states; see TRAIN silhouette and temporal proportions.")


# ======== CELL 61 [markdown] ========
# ## 20. Falsification Tests
# 
# Try to disprove the detector using NORMAL TEST windows under fixed references/cutoffs. Gain/offset change magnitude; shuffle destroys temporal ordering; circular shift changes phase. All three sensors use the same permutation/shift to preserve inter-channel pairing. Hann vs rectangular is a representation-mismatch probe plus normal-recalibrated comparison. Alternate references use early/late NORMAL TRAIN only.
# 

# ======== CELL 62 [code] ========
GAIN_TEST_FACTORS = [0.8,1.2]
OFFSET_TEST_TRAIN_STD_FRACTION = 0.2
FALSIFICATION_MAX_WINDOWS = 500
ALTERNATE_REFERENCE_BLOCKS = 2


# ======== CELL 63 [code] ========
def transform_and_score(base,transform):
    waves,records = {},[]
    for _,row in base.iterrows():
        values = transform(waveforms[row.window_id].copy(),row,window_times[row.window_id])
        waves[row.window_id]=values
        record = row.to_dict();record.update(physics_vector(values,window_times[row.window_id]));records.append(record)
    changed_features = pd.DataFrame(records)
    return score_components(changed_features,waves,window_times)

def sensitivity_metrics(changed,name):
    windows = alarm_windows(changed,FINAL_CANDIDATE,FINAL_POLICY)
    summary = segment_summary.loc[segment_summary.segment_id.isin(changed.segment_id)]
    segments = segment_decisions(windows,summary)
    return {"test":name,"n_windows":len(windows),"n_segments":len(segments),
        "warning_window_rate":windows.warning.mean(),"warning_segment_rate":segments.warning.mean(),
        "attention_segment_rate":segments.attention.mean(),"median_joint_score":changed.joint_score.median(),
        "median_current_score":changed.current_score.median(),
        "median_shape_score":changed.get("shape_mean_std_score",pd.Series(dtype=float)).median(),
        "median_spectrum_score":changed.spectrum_score.median()}

selected_ids = normal_test_summary.sort_values("segment_start").segment_id.tolist()
selected_frame = normal_test_scores.loc[normal_test_scores.segment_id.isin(selected_ids)].copy()
# Budget limit retains whole segments, never arbitrary truncations of alarm history.
counts = selected_frame.groupby("segment_id",sort=False).size().cumsum()
budget_ids = counts.loc[counts<=FALSIFICATION_MAX_WINDOWS].index
if not len(budget_ids): budget_ids=counts.index[:1]
falsification_base = selected_frame.loc[selected_frame.segment_id.isin(budget_ids)]
normal_train_sensor_std = train_rows[SENSOR_COLUMNS].std().to_numpy()
offset_vector = normal_train_sensor_std*OFFSET_TEST_TRAIN_STD_FRACTION
rng = np.random.default_rng(RANDOM_STATE)
transforms = {"unchanged":lambda x,row,t:x,
    **{f"gain_{gain}":lambda x,row,t,gain=gain:x*gain for gain in GAIN_TEST_FACTORS},
    "offset":lambda x,row,t:x+offset_vector,
    "sample_order_shuffle":lambda x,row,t:x[rng.permutation(len(x))],
    "circular_shift":lambda x,row,t:np.roll(x,max(1,len(x)//3),axis=0)}
falsification_records = []
for name,transform in transforms.items():
    changed = transform_and_score(falsification_base,transform)
    falsification_records.append(sensitivity_metrics(changed,name))

if spectral_references:
    n = max(spectral_references)
    reference = spectral_references[n]
    eligible = falsification_base.loc[falsification_base.window_n.eq(n)&falsification_base.uniform_sampling]
    cutoff_eligible = cutoff_rows.loc[cutoff_rows.window_n.eq(n)&cutoff_rows.uniform_sampling]
    train_eligible = physics_train.loc[physics_train.window_n.eq(n)&physics_train.uniform_sampling]
    rectangular_train = np.vstack([spectrum_vector(waveforms[wid],"rectangular") for wid in train_eligible.window_id])
    rectangular_reference = NearestNeighbors(n_neighbors=reference["k"]).fit(rectangular_train)
    for label,model,cutoff_mode in [("rectangular_against_hann",reference["model"],"frozen Hann"),
                                    ("rectangular_refitted_normal",rectangular_reference,"normal rectangular")]:
        if eligible.empty: continue
        matrix = np.vstack([spectrum_vector(waveforms[wid],"rectangular") for wid in eligible.window_id])
        values = model.kneighbors(matrix)[0].mean(axis=1)
        if cutoff_mode=="normal rectangular":
            calibration_matrix = np.vstack([spectrum_vector(waveforms[wid],"rectangular") for wid in cutoff_eligible.window_id])
            cutoff = np.quantile(model.kneighbors(calibration_matrix)[0].mean(axis=1),WARNING_QUANTILE)
        else: cutoff=thresholds["spectrum_score"][str(WARNING_QUANTILE)]
        segment_flag = eligible[["segment_id"]].assign(flag=values>cutoff).groupby("segment_id").flag.max()
        falsification_records.append({"test":label,"n_windows":len(values),"n_segments":len(segment_flag),
            "warning_window_rate":np.mean(values>cutoff),"warning_segment_rate":segment_flag.mean(),
            "median_spectrum_score":np.median(values),"branch":"spectral only","cutoff_mode":cutoff_mode})

train_segment_ids = physics_train.sort_values("start_time").segment_id.drop_duplicates().to_numpy()
for block_index,ids in enumerate(np.array_split(train_segment_ids,ALTERNATE_REFERENCE_BLOCKS)):
    block = physics_train.loc[physics_train.segment_id.isin(ids)]
    if len(block)<MIN_TRAIN_WINDOWS: continue
    reference = fit_mcd_reference(block,PHYSICS_FEATURES)
    cutoff = np.quantile(mcd_score(reference,cutoff_rows),WARNING_QUANTILE)
    distances = mcd_score(reference,normal_test_scores)
    flags = normal_test_scores[["segment_id"]].assign(flag=distances>cutoff).groupby("segment_id").flag.max()
    falsification_records.append({"test":f"normal_reference_block_{block_index}","n_windows":len(distances),
        "n_segments":len(flags),"warning_window_rate":np.mean(distances>cutoff),"warning_segment_rate":flags.mean(),
        "median_joint_score":np.median(distances),"branch":"joint threshold only"})
falsification_results = save_table(pd.DataFrame(falsification_records),"falsification_results.csv")
display(falsification_results)
confound_records = []
for split,group in scores.loc[scores.source.eq("normal")].groupby("split"):
    for column in thresholds:
        for metadata in ["gap_before_sec","segment_duration_sec","segment_n_samples","position_in_segment"]:
            pairs=group[[column,metadata]].dropna()
            correlation = stats.spearmanr(pairs[column],pairs[metadata]).statistic if len(pairs)>2 and pairs[metadata].nunique()>1 else np.nan
            confound_records.append({"split":split,"score":column,"qc_metadata":metadata,
                                     "spearman_correlation":correlation,"n_overlapping_windows":len(pairs)})
confound_audit = save_table(pd.DataFrame(confound_records),"confound_audit.csv")
display(confound_audit)
print("Confound correlations are descriptive; overlapping windows violate independent-sample significance assumptions.")
print("Gain/offset responses indicate magnitude dependence; shuffle/shift responses indicate temporal/phase dependence. Any instability is retained above.")


# ======== CELL 64 [markdown] ========
# ## 21. Synthetic Perturbation Sensitivity
# 
# Synthetic perturbations test a response to known mathematical changes in normal data. They are not real fault accuracy. All references and cutoffs remain fixed. Report both joint raw exceedance and the locked alarm response, exposing confirmation rules that suppress magnitude-only changes.
# 

# ======== CELL 65 [code] ========
SYNTHETIC_VIBRATION_GAIN = 1.5
SYNTHETIC_IMPULSE_STD_MULTIPLIER = 4.0
SYNTHETIC_CURRENT_CENTER_STD_FRACTION = 0.5
SYNTHETIC_CURRENT_AMPLITUDE_GAIN = 1.3
SYNTHETIC_DISTORTION_STD_FRACTION = 0.3


# ======== CELL 66 [code] ========
def perturb(values,row,t,kind):
    x=values.copy()
    if kind=="vibration_scale":x[:,:2]*=SYNTHETIC_VIBRATION_GAIN
    elif kind=="vibration_impulse":x[len(x)//2,:2]+=SYNTHETIC_IMPULSE_STD_MULTIPLIER*normal_train_sensor_std[:2]
    elif kind=="current_center_shift":x[:,2]+=SYNTHETIC_CURRENT_CENTER_STD_FRACTION*normal_train_sensor_std[2]
    elif kind=="current_amplitude_increase":x[:,2]=x[:,2].mean()+SYNTHETIC_CURRENT_AMPLITUDE_GAIN*(x[:,2]-x[:,2].mean())
    elif kind=="current_distortion":x[:,2]+=SYNTHETIC_DISTORTION_STD_FRACTION*normal_train_sensor_std[2]*np.sin(2*np.pi*2*t)
    else:raise ValueError(kind)
    return x

synthetic_records = []
SYNTHETIC_KINDS = ["vibration_scale","vibration_impulse","current_center_shift","current_amplitude_increase","current_distortion"]
synthetic_feature_sets = {}
for kind in SYNTHETIC_KINDS:
    changed = transform_and_score(falsification_base,lambda x,row,t,kind=kind:perturb(x,row,t,kind))
    synthetic_feature_sets[kind] = changed
    result = sensitivity_metrics(changed,kind)
    result["joint_raw_exceedance_rate"] = changed.joint_score.gt(thresholds["joint_score"][str(WARNING_QUANTILE)]).mean()
    result["metric_scope"] = "Synthetic perturbation sensitivity; not real fault accuracy"
    synthetic_records.append(result)
synthetic_sensitivity = save_table(pd.DataFrame(synthetic_records),"synthetic_sensitivity.csv")
display(synthetic_sensitivity)
print("Reported rates measure sensitivity to these chosen perturbation sizes, not detection of verified hydraulic/mechanical faults.")


# ======== CELL 67 [markdown] ========
# ## 22. Learning Curve
# 
# Only normal TRAIN segments enter fitted references. Prefixes preserve time order and whole segments. Recalibration uses the same normal cutoff subset; later normal and synthetic changes assess stability. Training fraction sufficiency and extreme-tail calibration sufficiency are different questions.
# 

# ======== CELL 68 [code] ========
LEARNING_CURVE_FRACTIONS = [0.05,0.10,0.25,0.50,1.00]


# ======== CELL 69 [code] ========
learning_records = []
for fraction in LEARNING_CURVE_FRACTIONS:
    ids = train_segment_ids[:max(1,int(np.ceil(len(train_segment_ids)*fraction)))]
    subset = physics_train.loc[physics_train.segment_id.isin(ids)]
    if len(subset)<max(MIN_TRAIN_WINDOWS,3*len(PHYSICS_FEATURES)):
        learning_records.append({"train_fraction":fraction,"train_windows":len(subset),"status":"insufficient covariance support"});continue
    reference = fit_mcd_reference(subset,PHYSICS_FEATURES)
    calibration_distances = mcd_score(reference,cutoff_rows)
    cutoff = np.quantile(calibration_distances,WARNING_QUANTILE)
    normal_distances = mcd_score(reference,normal_test_scores)
    normal_flags = normal_test_scores[["segment_id"]].assign(flag=normal_distances>cutoff).groupby("segment_id").flag.max()
    block_cutoffs = [np.quantile(block,WARNING_QUANTILE) for block in np.array_split(calibration_distances,3) if len(block)]
    synthetic_rates = [np.mean(mcd_score(reference,features)>cutoff) for features in synthetic_feature_sets.values()]
    learning_records.append({"train_fraction":fraction,"train_segments":len(ids),"train_windows":len(subset),"status":"fit",
        "effective_support":reference["diagnostics"]["effective_support"],"covariance_condition":reference["diagnostics"]["condition_after"],
        "normal_cutoff_q99":cutoff,"cutoff_block_cv":np.std(block_cutoffs)/max(np.mean(block_cutoffs),1e-12),
        "normal_test_false_segment_rate":normal_flags.mean(),"mean_synthetic_window_sensitivity":np.mean(synthetic_rates),
        "cutoff_calibration_windows":len(calibration_distances),"nominal_q99_tail_windows":len(calibration_distances)*.01,
        "branch":"joint reference threshold only"})
learning_curve = save_table(pd.DataFrame(learning_records),"learning_curve.csv")
display(learning_curve)
print("This curve concerns joint-reference estimation. It does not establish a small sample count sufficient for rare-tail alarms or independent conditions.")


# ======== CELL 70 [markdown] ========
# ## 23. Ablation & Final Model Selection
# 
# A–F are controlled representations and confirmation rules, not an expanding model zoo. Segment-level normal calibration determined the locked policy before TEST/abnormal evaluation. Retrospective sensitivity is now reported without changing that policy. Spectrum is kept as diagnostic evidence unless normal-only selection retained it.
# 

# ======== CELL 71 [code] ========
ABLATION_REPORT_POLICY = FINAL_POLICY


# ======== CELL 72 [code] ========
comparison_records,aggregation_records = [],[]
evaluation_scores = scores.loc[scores.split.isin(["normal_test","abnormal_evaluation"])]
evaluation_summary = segment_summary.loc[segment_summary.split.isin(["normal_test","abnormal_evaluation"])]
for candidate in candidate_definitions:
    for policy in ALARM_POLICIES:
        windows = alarm_windows(evaluation_scores,candidate,policy)
        segments = segment_decisions(windows,evaluation_summary)
        for level,frame in [("window",windows),("segment",segments)]:
            result = mixture_metrics(frame,level,candidate,policy)
            normal_group = segments.loc[segments.source.eq("normal")&segments.direct_evidence]
            result["normal_false_warning_segment_rate"] = normal_group.warning.mean()
            result["normal_attention_segment_rate"] = normal_group.attention.mean()
            result["abnormal_segment_recall_all"] = segments.loc[segments.source.eq("abnormal"),"warning"].mean()
            result["selected_pre_abnormal"] = candidate==FINAL_CANDIDATE and policy==FINAL_POLICY
            result["scope"] = "fixed retrospective comparator; no reselection"
            comparison_records.append(result)
    primary,_=candidate_definitions[candidate]
    for aggregation in SEGMENT_AGGREGATIONS:
        column=primary+"_"+aggregation
        threshold=SEGMENT_FRACTION_TRIGGER if aggregation=="fraction_above" else thresholds[primary][str(WARNING_QUANTILE)]
        frame=segment_scores.loc[segment_scores.split.isin(["normal_test","abnormal_evaluation"])].copy()
        frame["flag"]=frame[column]>threshold
        aggregation_records.append({"candidate":candidate,"aggregation":aggregation,
            "normal_false_segment_rate":frame.loc[frame.source.eq("normal")&frame[column].notna(),"flag"].mean(),
            "abnormal_recall_direct":frame.loc[frame.source.eq("abnormal")&frame[column].notna(),"flag"].mean(),
            "eligible_segments":int(frame[column].notna().sum()),"scope":"primary branch aggregation only"})
model_comparison = save_table(pd.DataFrame(comparison_records),"model_comparison.csv")
display(model_comparison.loc[model_comparison.level.eq("segment")])
display(save_table(pd.DataFrame(aggregation_records),"segment_aggregation_comparison.csv"))
# Fixed N=20/24 spectral comparison, independently calibrated from NORMAL only.
spectral_evaluation_records=[]
for n,frame in spectral_frames.items():
    cal=frame.loc[frame.calibration_role.eq("cutoff_fit"),"spectrum_score"]
    threshold=float(np.quantile(cal,WARNING_QUANTILE))
    evaluation=frame.loc[frame.split.isin(["normal_test","abnormal_evaluation"])].copy()
    evaluation["flag"]=evaluation.spectrum_score>threshold
    for split,group in evaluation.groupby("split"):
        spectral_evaluation_records.append({"n":n,"k":spectral_references[n]["k"],"split":split,
            "window_flag_rate":group.flag.mean(),"segment_flag_rate":group.groupby("segment_id").flag.max().mean(),
            "n_eligible_segments":group.segment_id.nunique(),"cutoff":threshold})
display(save_table(pd.DataFrame(spectral_evaluation_records,columns=["n","k","split","window_flag_rate","segment_flag_rate","n_eligible_segments","cutoff"]),"spectral_window_comparison.csv"))
spike_comparison_records=[]
for frame in [normal_analysis,abnormal_analysis]:
    features_current,_,_=make_windows(frame)
    raw_frame=frame.copy();raw_frame["AI2_analysis"]=raw_frame[SENSOR_COLUMNS[2]]
    features_raw,_,_=make_windows(raw_frame)
    proposed_frame,_=correct_isolated_spikes(raw_frame,apply_correction=True)
    features_proposed,_,_=make_windows(proposed_frame)
    for column in CURRENT_FEATURES:
        before=mcd_score(current_reference,features_raw)
        after=mcd_score(current_reference,features_current)
        proposed=mcd_score(current_reference,features_proposed)
        spike_comparison_records.append({"source":frame.source.iloc[0],"feature":column,
            "median_feature_before":features_raw[column].median(),"median_feature_after":features_current[column].median(),
            "median_abs_feature_change":np.median(np.abs(features_raw[column]-features_current[column])),
            "median_current_score_before":np.median(before),"median_current_score_after":np.median(after),
            "median_current_score_proposed_correction":np.median(proposed),
            "median_abs_proposed_feature_change":np.median(np.abs(features_raw[column]-features_proposed[column])),
            "correction_enabled":ENABLE_CURRENT_SPIKE_CORRECTION})
display(save_table(pd.DataFrame(spike_comparison_records),"current_spike_before_after.csv"))
print("Final policy remains",FINAL_CANDIDATE,FINAL_POLICY,"as selected before abnormal evaluation.")
print("Spectrum is", "part of selected confirmation" if "spectrum" in FINAL_CANDIDATE else "diagnostic only; not forced into final warnings", ".")
print("Selection is constrained by normal-only evidence; abnormal sensitivity may be insufficient and is disclosed, not optimized away.")
selected_primary=model_comparison.loc[model_comparison.candidate.eq(FINAL_CANDIDATE)&model_comparison.level.eq("segment")]
print("Persistence comparison for the selected representation:")
display(selected_primary[["policy","normal_false_warning_segment_rate","recall_direct","abnormal_segment_recall_all"]])


# ======== CELL 73 [markdown] ========
# ## 24. Final Visualizations
# 
# Each chart is a standalone matplotlib figure. Scores use sensor/deviation units, not mechanical fault probabilities. Raw examples are separate recordings; horizontal sample/time alignment is a visual comparison only.
# 

# ======== CELL 74 [code] ========
RAW_EXAMPLE_MAX_SAMPLES = 50
SCORE_FIGURE_LOG1P = True


# ======== CELL 75 [code] ========
for column in ["shape_mean_std_score","shape_median_mad_score","spectrum_score"]:
    if column not in thresholds:
        (FIGURE_DIR/f"score_distribution_{column}.png").unlink(missing_ok=True)
if not ("shape_mean_std_score" in thresholds and "spectrum_score" in thresholds):
    (FIGURE_DIR/"shape_vs_spectrum_score.png").unlink(missing_ok=True)
for channel,sensor in enumerate(SENSOR_COLUMNS):
    plt.figure()
    for frame,label in [(normal_analysis,"Normal"),(abnormal_analysis,"Recorded abnormal")]:
        eligible_segments=frame.groupby("segment_id").size().sort_values(ascending=False)
        example=frame.loc[frame.segment_id.eq(eligible_segments.index[0])].iloc[:RAW_EXAMPLE_MAX_SAMPLES]
        relative=(example.timestamp-example.timestamp.iloc[0]).dt.total_seconds()
        plt.plot(relative,example[sensor],label=label,alpha=.8)
    plt.xlabel("Relative time within each selected recording segment (seconds)");plt.ylabel("Sensor value (unit unverified)")
    plt.title(f"Representative raw {sensor}");plt.legend();save_figure(f"representative_raw_{sensor}.png")

for feature in PHYSICS_FEATURES:
    plt.figure()
    groups=[physics.loc[physics.split.eq(split),feature].to_numpy() for split in ["train","calibration","normal_test","abnormal_evaluation"]]
    plt.boxplot(groups,tick_labels=["Train","Calibration","Normal test","Abnormal"],showfliers=False)
    plt.ylabel(feature+" (sensor-derived units)");plt.title("Physics feature distribution: "+feature)
    save_figure("physics_feature_"+feature+".png")

for column in thresholds:
    plt.figure()
    for split,group in scores.groupby("split",sort=False):
        values=group[column].dropna()
        if len(values):plt.hist(np.log1p(values) if SCORE_FIGURE_LOG1P else values,bins=40,density=True,alpha=.4,label=split)
    plt.xlabel(("log(1 + " if SCORE_FIGURE_LOG1P else "")+column+(")" if SCORE_FIGURE_LOG1P else ""))
    plt.ylabel("Relative density");plt.title("Anomaly evidence by chronological partition");plt.legend()
    save_figure("score_distribution_"+column+".png")
plt.figure()
for split,group in scores.groupby("split",sort=False):
    plt.scatter(np.log1p(group.joint_score),np.log1p(group.current_score),s=10,alpha=.4,label=split)
plt.xlabel("log(1 + joint squared distance)");plt.ylabel("log(1 + current squared distance)")
plt.title("Joint vs current evidence (not averaged)");plt.legend();save_figure("joint_vs_current_score.png")

if "shape_mean_std_score" in thresholds and "spectrum_score" in thresholds:
    plt.figure()
    for split,group in scores.groupby("split",sort=False):
        plt.scatter(np.log1p(group.shape_mean_std_score),group.spectrum_score,s=10,alpha=.4,label=split)
    plt.xlabel("log(1 + normalized waveform distance)");plt.ylabel("Low-frequency spectrum kNN distance")
    plt.title("Independent waveform / spectral evidence");plt.legend();save_figure("shape_vs_spectrum_score.png")

threshold_sensitivity_records=[]
for column in thresholds:
    for q,cutoff in thresholds[column].items():
        for split in ["calibration","normal_test","abnormal_evaluation"]:
            frame=scores.loc[scores.split.eq(split)&scores[column].notna()]
            flagged=frame[["segment_id"]].assign(flag=frame[column]>cutoff).groupby("segment_id").flag.max()
            threshold_sensitivity_records.append({"detector":column,"quantile":float(q),"cutoff":cutoff,"split":split,
                "window_exceedance_rate":frame[column].gt(cutoff).mean(),"segment_exceedance_rate":flagged.mean(),
                "n_eligible_segments":len(flagged),"cutoff_source":"normal calibration cutoff_fit"})
threshold_sensitivity=save_table(pd.DataFrame(threshold_sensitivity_records),"threshold_sensitivity.csv")
plt.figure()
for split,group in threshold_sensitivity.loc[threshold_sensitivity.detector.eq("joint_score")].groupby("split"):
    plt.plot(group["quantile"],group.segment_exceedance_rate,"o-",label=split)
plt.xlabel("Normal calibration cutoff quantile");plt.ylabel("Segment exceedance rate")
plt.title("Frozen threshold sensitivity (no abnormal optimization)");plt.legend();save_figure("threshold_sensitivity.png")

for source in ["normal","abnormal"]:
    plt.figure()
    for sid,group in scores.loc[scores.source.eq(source)].groupby("segment_id",sort=False):
        plt.plot(group.available_at,np.log1p(group.joint_score),color="steelblue",linewidth=.7)
    plt.axhline(np.log1p(thresholds["joint_score"][str(WARNING_QUANTILE)]),color="tomato",linestyle="--",label="Normal q99")
    plt.xlabel("Actual window completion timestamp");plt.ylabel("log(1 + joint distance)")
    plt.title(f"{source}: gap-aware anomaly evidence over actual time");plt.legend();save_figure(f"score_over_timestamp_{source}.png")

plt.figure()
for candidate,group in regime_evaluation.groupby("candidate",sort=False):
    plt.plot(group.regime,group.warning_rate,"o-",label=candidate)
plt.xlabel("Descriptive normal regime ID");plt.ylabel("Normal test false-warning segment rate")
plt.title("False warnings across normal operating regimes");plt.legend(fontsize=8);save_figure("false_alarms_by_regime.png")

plt.figure()
group=model_comparison.loc[model_comparison.policy.eq(FINAL_POLICY)]
for level,frame in group.groupby("level"):
    plt.plot(frame.candidate,frame.recall_direct,"o-",label=level)
plt.xticks(rotation=35,ha="right");plt.ylim(0,1.05);plt.ylabel("Recorded abnormal recall among eligible evidence")
plt.title("Window vs acquisition-segment retrospective results");plt.legend();save_figure("window_vs_segment_results.png")
plt.figure()
for level,frame in group.groupby("level"):
    false_rate=frame.false_positive/(frame.false_positive+frame.true_negative)
    plt.plot(frame.candidate,false_rate,"o-",label=level)
plt.xticks(rotation=35,ha="right");plt.ylim(0,1.05);plt.ylabel("Later-normal false-warning rate among eligible evidence")
plt.title("Window vs segment false-warning burden");plt.legend();save_figure("window_vs_segment_false_warning.png")
plt.figure();plt.bar(falsification_results.test,falsification_results.warning_segment_rate)
plt.xticks(rotation=40,ha="right");plt.ylabel("Normal-derived segment warning / exceedance rate")
plt.title("Falsification: keep adverse responses visible");save_figure("falsification_summary.png")
plt.figure()
fit_curve=learning_curve.loc[learning_curve.status.eq("fit")]
plt.plot(fit_curve.train_fraction,fit_curve.normal_test_false_segment_rate,"o-",label="Later-normal segment exceedance")
plt.plot(fit_curve.train_fraction,fit_curve.mean_synthetic_window_sensitivity,"o-",label="Synthetic window sensitivity")
plt.xlabel("Chronological normal TRAIN segment fraction");plt.ylabel("Rate")
plt.title("Joint-reference learning curve; threshold support is separate");plt.legend();save_figure("learning_curve.png")
plt.figure()
for block,group in block_table.groupby("temporal_block"):
    plt.hist(np.log1p(group.joint_score),bins=25,density=True,alpha=.4,label=f"Normal test block {block}")
plt.xlabel("log(1 + joint distance)");plt.ylabel("Relative density");plt.title("Later-normal temporal block stability")
plt.legend();save_figure("normal_temporal_block_scores.png")
plt.figure()
for regime,group in normal_test_scores.groupby("regime"):
    plt.hist(np.log1p(group.joint_score),bins=25,density=True,alpha=.4,label=f"Regime {regime}")
plt.xlabel("log(1 + joint distance)");plt.ylabel("Relative density");plt.title("Normal regime score distributions")
plt.legend();save_figure("normal_regime_score_distributions.png")
print("Saved standalone figures:",len(list(FIGURE_DIR.glob("*.png"))))


# ======== CELL 76 [markdown] ========
# ## 25. Competition Submission Export
# 
# Inspect an optional sample submission; its unknown target/id columns cannot be inferred safely. Edit the mapping below. Row coverage uses completed window decisions assigned to their contributing original rows (retrospective export, not causal row-at-time detection). Exact duplicates inherit the canonical measurement decision. Uncovered rows are marked INSUFFICIENT; optional nearest-window propagation stays within the same segment and is labeled PROPAGATED. <8 segments never acquire direct detections.
# 

# ======== CELL 77 [code] ========
SUBMISSION_TARGET_COLUMN = "prediction"  # edit to the actual competition target
SUBMISSION_ID_COLUMN = "row_id"  # set None if absent from actual format
SAMPLE_SUBMISSION_ID_COLUMN = None  # if set, align sample IDs with source row IDs; assert bijection
SUBMISSION_PROPAGATE_UNCOVERED = False
SUBMISSION_UNSCORED_LABEL = 0  # export-only assumption, not confirmed normal evidence
SUBMISSION_POSITIVE_LABEL = 1
SUBMISSION_NEGATIVE_LABEL = 0


# ======== CELL 78 [code] ========
if SUBMISSION_INPUT_FILE is not None:
    submission_path=resolve_input(SUBMISSION_INPUT_FILE)
    submission_raw=read_sensor_csv(submission_path,"submission",-1)
    # Unlabeled inference: audit without the normal/abnormal expected-label check.
    if LABEL_COLUMN in submission_raw:submission_raw=submission_raw.drop(columns=LABEL_COLUMN)
    submission_clean,submission_map,submission_audit,submission_conflicts=audit_and_clean(submission_raw)
    submission_clean["dt_sec"]=submission_clean.timestamp.diff().dt.total_seconds()
    submission_processed,submission_segments=construct_segments(submission_clean,SEGMENT_GAP_SEC)
    submission_processed["split"]="submission";submission_processed["calibration_role"]="not_calibration"
    submission_segments["split"]="submission";submission_segments["calibration_role"]="not_calibration";submission_segments["regime"]=-1
    submission_analysis,submission_spikes=correct_isolated_spikes(submission_processed)
    submission_features,submission_waves,submission_times=make_windows(submission_analysis)
    if len(submission_features):
        submission_scores=score_components(submission_features,submission_waves,submission_times)
        submission_windows=alarm_windows(submission_scores,FINAL_CANDIDATE,FINAL_POLICY)
    else:submission_windows=final_windows.iloc[:0].copy()
    save_table(pd.DataFrame([submission_audit]),"submission_input_audit.csv")
else:
    submission_raw=abnormal_raw.copy();submission_map=abnormal_row_map.copy()
    submission_processed=abnormal.copy();submission_windows=abnormal_final_windows.copy()
    print("Export target is supplied abnormal recording for demonstration; set SUBMISSION_INPUT_FILE for an unlabeled competition file.")

def map_to_original_rows(raw,mapping,processed,windows):
    canonical_records=[]
    for _,row in processed.iterrows():
        relevant=windows.loc[windows.row_ids.map(lambda ids:int(row.source_row_id) in ids)]
        eligible=relevant.loc[relevant.candidate_evidence_available]
        direct=len(eligible)>0
        propagated=False
        warning=bool(eligible.warning.any()) if direct else False
        attention=bool(eligible.attention.any()) if direct else False
        # A positive label exists at the first positive window completion. A negative OR label
        # is final only after all covering windows complete; never timestamp a later warning early.
        availability=(eligible.loc[eligible.warning,"available_at"].min() if warning
                      else eligible.available_at.max()) if direct else pd.NaT
        if not direct and SUBMISSION_PROPAGATE_UNCOVERED and row.valid_measurement:
            same_segment=windows.loc[windows.segment_id.eq(row.segment_id)&windows.candidate_evidence_available]
            if len(same_segment):
                nearest=(same_segment.available_at-row.timestamp).abs().idxmin()
                warning=bool(same_segment.loc[nearest,"warning"]);attention=bool(same_segment.loc[nearest,"attention"])
                availability=same_segment.loc[nearest,"available_at"];propagated=True
        evidence="DIRECT" if direct else ("PROPAGATED" if propagated else "INSUFFICIENT")
        exported=SUBMISSION_POSITIVE_LABEL if warning else SUBMISSION_NEGATIVE_LABEL
        if evidence=="INSUFFICIENT":exported=SUBMISSION_UNSCORED_LABEL
        canonical_records.append({"source":row.source,"canonical_row_id":int(row.source_row_id),
            "segment_id":row.segment_id,"prediction":exported,"warning_evidence":warning,"attention_evidence":attention,
            "evidence_status":evidence,"available_at":availability,"valid_measurement":row.valid_measurement})
    result=mapping.merge(pd.DataFrame(canonical_records),on=["source","canonical_row_id"],how="left",validate="many_to_one")
    result=result.sort_values("source_row_id",kind="stable").reset_index(drop=True)
    assert len(result)==len(raw) and result.source_row_id.to_list()==list(range(len(raw)))
    assert result.evidence_status.notna().all() and result.prediction.notna().all()
    return result

row_predictions=map_to_original_rows(submission_raw,submission_map,submission_processed,submission_windows)
save_table(row_predictions,"submission_row_evidence.csv")
sample_path=resolve_input(SAMPLE_SUBMISSION_FILE)
if sample_path is not None:
    sample=pd.read_csv(sample_path)
    print("Sample submission shape/columns:",sample.shape,list(sample));display(sample.head())
    if SUBMISSION_TARGET_COLUMN not in sample:raise ValueError("Edit SUBMISSION_TARGET_COLUMN to match sample submission")
    assert len(sample)==len(row_predictions),"Sample/input row count mismatch; provide correct submission input"
    submission=sample.copy()
    if SAMPLE_SUBMISSION_ID_COLUMN:
        if SAMPLE_SUBMISSION_ID_COLUMN not in sample:raise ValueError("Configured sample ID column missing")
        assert sample[SAMPLE_SUBMISSION_ID_COLUMN].is_unique
        assert set(sample[SAMPLE_SUBMISSION_ID_COLUMN])==set(row_predictions.source_row_id),"Map actual IDs explicitly before export"
        predicted_by_id=row_predictions.set_index("source_row_id").prediction
        submission[SUBMISSION_TARGET_COLUMN]=sample[SAMPLE_SUBMISSION_ID_COLUMN].map(predicted_by_id)
    else:
        print("Assumption: sample submission order equals original input CSV order. Verify against competition instructions.")
        submission[SUBMISSION_TARGET_COLUMN]=row_predictions.prediction.to_numpy()
else:
    submission=pd.DataFrame({SUBMISSION_TARGET_COLUMN:row_predictions.prediction.to_numpy()})
    if SUBMISSION_ID_COLUMN:submission.insert(0,SUBMISSION_ID_COLUMN,row_predictions.source_row_id.to_numpy())
    print("No sample submission: editable demonstration format; official schema remains unverified.")
assert len(submission)==len(submission_raw) and submission[SUBMISSION_TARGET_COLUMN].notna().all()
submission.to_csv(OUTPUT_DIR/"submission.csv",index=False,encoding="utf-8-sig")
label_scope_summary=row_predictions.groupby("evidence_status").agg(rows=("source_row_id","size"),positive_labels=("warning_evidence","sum")).reset_index()
display(save_table(label_scope_summary,"direct_vs_propagated_labels.csv"))
print("All original rows preserved:",len(submission),"; DIRECT and PROPAGATED counts are separate.")


# ======== CELL 79 [markdown] ========
# ## 26. Save Models & Artifacts
# 
# Save fitted scalers, references, separate thresholds, waveform/spectral branches, regime model and feature names. Record input/config hashes and the frozen selection. Intermediate CSVs and joblib snapshots let you inspect or restart expensive stages. Only load trusted notebook-produced joblib files.
# 

# ======== CELL 80 [code] ========
SAVE_INTERMEDIATE_WAVEFORMS = True


# ======== CELL 81 [code] ========
joblib.dump(joint_reference,MODEL_DIR/"joint_mcd.joblib")
joblib.dump(current_reference,MODEL_DIR/"current_mcd.joblib")
joblib.dump({"joint":joint_reference["scaler"],"current":current_reference["scaler"],"regime":regime_scaler},MODEL_DIR/"scalers.joblib")
joblib.dump(waveform_references,MODEL_DIR/"waveform_models.joblib")
joblib.dump(spectral_references,MODEL_DIR/"spectral_knn.joblib")
joblib.dump({"model":regime_model,"scaler":regime_scaler,"features":REGIME_FEATURES,
             "regimes_are_fault_labels":False},MODEL_DIR/"kmeans_regimes.joblib")
joblib.dump({"thresholds":thresholds,"ewma_thresholds":ewma_thresholds,"alpha":EWMA_ALPHA,
             "candidate_definitions":candidate_definitions,"selection":FINAL_SELECTION_LOCK},MODEL_DIR/"alarm_policy.joblib")
write_json(MODEL_DIR/"feature_names.json",{"physics":PHYSICS_FEATURES,"current":CURRENT_FEATURES,
    "shape":"per-window normalized 3-channel waveform","spectral":"normalized DC-excluded current power"})
if SAVE_INTERMEDIATE_WAVEFORMS:
    joblib.dump({"waveforms":waveforms,"times":window_times},CACHE_DIR/"window_waveforms.joblib")
RUN_CONFIG=config_snapshot();RUN_CONFIG["input_sha256"]=INPUT_HASHES
RUN_CONFIG["final_selection"]=FINAL_SELECTION_LOCK
write_json(OUTPUT_DIR/"run_config.json",RUN_CONFIG)
required_tables=["data_audit.csv","segment_summary.csv","split_summary.csv","regime_summary.csv","feature_summary.csv",
    "threshold_sensitivity.csv","model_comparison.csv","segment_predictions.csv","alarm_episodes.csv","confound_audit.csv",
    "falsification_results.csv","synthetic_sensitivity.csv","learning_curve.csv","current_spike_audit.csv"]
assert all((TABLE_DIR/name).is_file() for name in required_tables)
artifact_manifest={"files":[{"path":str(path.relative_to(OUTPUT_DIR)),"bytes":path.stat().st_size,
    "sha256":hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(OUTPUT_DIR.rglob("*")) if path.is_file()
    and path.name!="artifact_manifest.json"],"input_sha256":INPUT_HASHES,"selection":FINAL_SELECTION_LOCK,
    "verification":{"normal_segments_not_split":True,"normal_only_fitting":True,"normal_only_cutoffs":True,
                    "no_cross_segment_windows":True,"submission_all_rows_preserved":True}}
write_json(OUTPUT_DIR/"artifact_manifest.json",artifact_manifest)
print("Saved required tables:",len(required_tables),"models:",len(list(MODEL_DIR.glob("*"))),"submission rows:",len(submission))
print("Expensive feature and MCD caches use fingerprint checks; lightweight tables and plots are regenerated transparently.")


# ======== CELL 82 [markdown] ========
# ## 27. Final Results Summary
# 
# Automatically distinguish observed results, assumptions, retrospective evaluation and synthetic experiments. No unsupported prognostic claim is generated. A poor result remains visible and is not replaced by an abnormal-optimized pipeline.
# 

# ======== CELL 83 [code] ========
FINAL_REPORT_FILENAME = "FINAL_RESULTS_SUMMARY.md"


# ======== CELL 84 [code] ========
direct_abnormal_count=int(abnormal_final_segments.direct_evidence.sum())
warning_abnormal_count=int(abnormal_final_segments.warning.sum())
normal_warning_count=int(normal_final_segments.warning.sum())
summary_text=f'''# KAMP normal-reference competition results

## Confirmed structure and methodology
- Raw normal rows: {len(normal_raw)}; raw abnormal rows: {len(abnormal_raw)}.
- Exact duplicates removed: normal {normal_audit['exact_duplicate_rows']}, abnormal {abnormal_audit['exact_duplicate_rows']}.
- Observed normal median dt: {NORMAL_MEDIAN_DT:.6g}s ({OBSERVED_FS_HZ:.6g}Hz); Nyquist {OBSERVED_FS_HZ/2:.6g}Hz.
- Gap rule: dt > {SEGMENT_GAP_SEC}s; duplicate/nonpositive dt and invalid rows also isolate segments.
- Acquisition segments are not established physical press cycles.
- Normal chronological split ratios by segment count: {NORMAL_SPLIT_RATIOS}; windows created after the split.
- Nine physics features: {', '.join(PHYSICS_FEATURES)}.
- MCD effective support: joint {joint_reference['diagnostics']['effective_support']}, current {current_reference['diagnostics']['effective_support']}.
- All reference fitting and scaling: NORMAL TRAIN only. Abnormal used for neither fitting nor selection.
- Final frozen candidate/policy: **{FINAL_CANDIDATE} / {FINAL_POLICY}**.
- Cutoffs: NORMAL CALIBRATION cutoff-fit subset; Attention q{100*ATTENTION_QUANTILE:g}, Warning q{100*WARNING_QUANTILE:g}.
- Policy selected using separate normal calibration segments before TEST/abnormal evaluation.

## Temporal normal results
- Warning segments: {normal_warning_count}/{len(normal_final_segments)} all supplied normal-test segments.
- Direct-evidence false-warning segment rate: {final_normal_result.normal_false_warning_segment_rate:.4f}.
- Direct-evidence Attention-or-Warning segment rate: {final_normal_result.normal_attention_segment_rate:.4f}.
- Alarm episodes: {len(normal_episodes)}; maximum consecutive warning-window run: {int(final_normal_result.maximum_consecutive_false_positive_run)}.
- Regime false-warning disparity: {regime_disparity:.4f}; TRAIN silhouette: {regime_silhouette:.4f}.

## Retrospective recorded-abnormal evaluation
- Warning segments: {warning_abnormal_count}/{direct_abnormal_count} eligible segments.
- Direct recall: {abnormal_segment_recall_direct:.4f}; all-recorded-segment recall: {abnormal_segment_recall_all:.4f}.
- {len(abnormal_final_segments)-direct_abnormal_count} segments lack direct final-candidate evidence.
- One abnormal-state recording: segments/windows are not independent fault episodes.
- Precision/F1/PR-AUC refer to the constructed recording mixture, not field prevalence.
- Submission input rows: {len(submission_raw)}; all original row order preserved; evidence status in tables/submission_row_evidence.csv.

## Assumptions and unresolved acquisition questions
- Sensor physical units, calibration, anti-aliasing bandwidth and instantaneous-current interpretation are unverified.
- 0.6Hz is assumed; normal N=24 scan best={best_scan_frequency:g}Hz; assumption reproduced={SINE_ASSUMPTION_SUPPORTED}.
- Current-fit confidence by N: {current_confidence}; conditional iid residual bootstrap has dependent-sample limitations.
- Spike correction enabled={ENABLE_CURRENT_SPIKE_CORRECTION}; candidates={len(spike_audit)}. Changes are audited.
- KMeans IDs are descriptive operating regimes, never fault labels. Stable physical regime meaning is unverified.
- <8-sample segments have insufficient direct physics evidence. Propagation enabled={SUBMISSION_PROPAGATE_UNCOVERED} is export-only.
- Official competition schema is unverified unless the mapping was checked against its sample/instructions.

## Synthetic sensitivity and falsification
- Mean synthetic segment warning sensitivity: {synthetic_sensitivity.warning_segment_rate.mean():.4f}; mathematical perturbations only.
- Gain/offset/order/phase/taper/reference-block probes: tables/falsification_results.csv; adverse responses retained.
- QC confound correlations: tables/confound_audit.csv; overlapping samples prevent independent p-value interpretation.
- Learning curve measures covariance estimation separately from rare-tail cutoff sufficiency; no universal minimum data claim.
- Magnitude features can respond to gain/load changes; shape confirmation can suppress magnitude-only perturbations.

## Limits and defensible claim
Normal/abnormal date and acquisition conditions are confounded. No healthy→degradation→failure trajectory,
verified mechanical cause, independent failure episodes, remaining useful life, or real early-warning lead time is established.
Temporal normal TEST is within the supplied normal recording. Sparse calibration tails and dependent windows limit confidence.
Existing project exploration has seen these files; this notebook's locked protocol is not a new independent blind study.

Under the limited KAMP dataset, we constructed a normal-reference detector using interpretable vibration/current
features and robust statistical distance. Normal operating-regime discovery was separated from anomaly evidence,
false warnings were evaluated on chronologically later normal segments, and low-frequency spectral information
was treated as optional evidence. The provided abnormal recording was used only for retrospective evaluation.
These results characterize recorded-state deviation and synthetic sensitivity; they do not verify pump failure prediction.
'''
summary_text += "\n## Observed falsification responses (frozen settings)\n"
for test in ["unchanged","gain_0.8","gain_1.2","offset","sample_order_shuffle","circular_shift"]:
    rows=falsification_results.loc[falsification_results.test.eq(test)]
    if len(rows):summary_text+=f"- {test}: normal-derived segment warning rate {rows.iloc[0].warning_segment_rate:.4f}.\n"
summary_text += "\nOffset/phase sensitivity means statistical outlier robustness is not acquisition invariance. "
summary_text += "Circular wrapping introduces a boundary discontinuity; Hann taper can also change spectral shape after shifting. "
summary_text += "These controlled changes do not establish a mechanical fault.\n"
(OUTPUT_DIR/FINAL_REPORT_FILENAME).write_text(summary_text,encoding="utf-8")
write_json(OUTPUT_DIR/"run_config.json",config_snapshot())
artifact_manifest["files"]=[{"path":str(path.relative_to(OUTPUT_DIR)),"bytes":path.stat().st_size,
    "sha256":hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(OUTPUT_DIR.rglob("*"))
    if path.is_file() and path.name!="artifact_manifest.json"]
write_json(OUTPUT_DIR/"artifact_manifest.json",artifact_manifest)
display(Markdown(summary_text))
print("Persistent outputs:",OUTPUT_DIR)
print("Done. Review adverse robustness results and official submission mapping before submitting.")
