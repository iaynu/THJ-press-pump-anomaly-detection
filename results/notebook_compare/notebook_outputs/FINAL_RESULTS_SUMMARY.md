# KAMP normal-reference competition results

## Confirmed structure and methodology
- Raw normal rows: 20000; raw abnormal rows: 600.
- Exact duplicates removed: normal 0, abnormal 0.
- Observed normal median dt: 0.1s (10Hz); Nyquist 5Hz.
- Gap rule: dt > 0.5s; duplicate/nonpositive dt and invalid rows also isolate segments.
- Acquisition segments are not established physical press cycles.
- Normal chronological split ratios by segment count: [0.6, 0.2, 0.2]; windows created after the split.
- Nine physics features: AI0_rms, AI1_rms, AI0_abs_peak, AI1_abs_peak, sum_rms, difference_rms, current_center, current_amplitude, current_distortion.
- MCD effective support: joint 816, current 853.
- All reference fitting and scaling: NORMAL TRAIN only. Abnormal used for neither fitting nor selection.
- Final frozen candidate/policy: **A_physics / none**.
- Cutoffs: NORMAL CALIBRATION cutoff-fit subset; Attention q97.5, Warning q99.
- Policy selected using separate normal calibration segments before TEST/abnormal evaluation.

## Temporal normal results
- Warning segments: 3/120 all supplied normal-test segments.
- Direct-evidence false-warning segment rate: 0.0263.
- Direct-evidence Attention-or-Warning segment rate: 0.0877.
- Alarm episodes: 3; maximum consecutive warning-window run: 1.
- Regime false-warning disparity: 0.0312; TRAIN silhouette: 0.4973.

## Retrospective recorded-abnormal evaluation
- Warning segments: 18/18 eligible segments.
- Direct recall: 1.0000; all-recorded-segment recall: 0.8571.
- 3 segments lack direct final-candidate evidence.
- One abnormal-state recording: segments/windows are not independent fault episodes.
- Precision/F1/PR-AUC refer to the constructed recording mixture, not field prevalence.
- Submission input rows: 600; all original row order preserved; evidence status in tables/submission_row_evidence.csv.

## Assumptions and unresolved acquisition questions
- Sensor physical units, calibration, anti-aliasing bandwidth and instantaneous-current interpretation are unverified.
- 0.6Hz is assumed; normal N=24 scan best=0.6Hz; assumption reproduced=True.
- Current-fit confidence by N: {8: True, 16: True, 24: True}; conditional iid residual bootstrap has dependent-sample limitations.
- Spike correction enabled=False; candidates=3. Changes are audited.
- KMeans IDs are descriptive operating regimes, never fault labels. Stable physical regime meaning is unverified.
- <8-sample segments have insufficient direct physics evidence. Propagation enabled=False is export-only.
- Official competition schema is unverified unless the mapping was checked against its sample/instructions.

## Synthetic sensitivity and falsification
- Mean synthetic segment warning sensitivity: 0.4596; mathematical perturbations only.
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

## Observed falsification responses (frozen settings)
- unchanged: normal-derived segment warning rate 0.0263.
- gain_0.8: normal-derived segment warning rate 0.0000.
- gain_1.2: normal-derived segment warning rate 0.0702.
- offset: normal-derived segment warning rate 1.0000.
- sample_order_shuffle: normal-derived segment warning rate 1.0000.
- circular_shift: normal-derived segment warning rate 0.9123.

Offset/phase sensitivity means statistical outlier robustness is not acquisition invariance. Circular wrapping introduces a boundary discontinuity; Hann taper can also change spectral shape after shifting. These controlled changes do not establish a mechanical fault.
