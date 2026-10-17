#!/usr/bin/env bash
# Full pipeline. Place df_FL_CICIoT2023_6_classes.csv in data/ first (see data/README.md).
# Steps 03 and 04 are the expensive ones (several hours on a 4-core machine; all steps cache or save as they go).
set -e
python scripts/00_prepare_data.py
python scripts/01_reproduce_original.py
python scripts/02_label_verification.py
python scripts/03_leakage_audit.py
for m in Dummy LR MLP DT RF LGBM XGB; do python scripts/04_detection_models.py $m; done
python scripts/05_phase2_stages.py
python scripts/06_shap.py
python scripts/07_statistics.py
# External evaluation (optional; needs the cleaned full CICIoT2023 release, see external/README.md)
# python external/build_external_sample.py path/to/ciciot2023.parquet
# python external/external_evaluation.py
for f in figures/fig1_methodology.py figures/fig_leakage.py figures/fig_models_robustness.py figures/fig_shap.py figures/fig_external.py; do python $f; done
