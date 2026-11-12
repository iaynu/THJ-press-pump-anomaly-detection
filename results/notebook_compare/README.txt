Comparison with TalkFile_KAMP_Final_Competition_Pipeline_Colab.ipynb
====================================================================
runner/run_nb.py        the notebook's cells in order, with only two edits: PROJECT_ROOT -> a local folder, USE_GOOGLE_DRIVE=False
runner/nb_side.py       runs inside the notebook's namespace after it finished; scores OUR 11 injected faults with ITS final model
notebook_outputs/       the notebook's own result tables (as produced by the run) + the two side files
ours_*.csv              our models under the notebook's protocol (python -m kamp.experiments.notebook_compare)
01..04_*.csv, F8_*.png  side-by-side tables (python -m kamp.experiments.notebook_compare_report)

Same protocol: normal segments in time order, first 60% train / next 20% calibration / last 20% test (120 segments, 114 with >=8 rows).
Our models are fitted on the same train segments only; our own out-of-fold calibration replaces the notebook's calibration subset.
Each model uses its own warning rule (notebook: window q99 from its calibration subset, segment warns if any window warns;
ours: segment-level fused q99 from out-of-fold normal scores). AUC columns are threshold-free.
