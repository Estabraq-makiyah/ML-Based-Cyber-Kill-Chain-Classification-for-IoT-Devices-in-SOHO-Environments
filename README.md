# Stage-Aware Intrusion Classification for SOHO IoT Traffic under Leakage-Controlled Evaluation

Code and result files for the article by Benjamin Aziz, Estabraq Makiyah and Ukamaka Oragwu.
Every number, table and figure in the revised manuscript is produced by the scripts below.

## Pipeline

| Step | Script | Paper section | Output |
|---|---|---|---|
| 0 | `scripts/00_prepare_data.py` | 3.1, 3.2 | `data/data.parquet`, `data/dedup.parquet` (unique records + capture-time block) |
| 1 | `scripts/01_reproduce_original.py` | 4.1 | Original random-split results; test rows with an exact copy in training; Phase 2 relabelling check |
| 2 | `scripts/02_label_verification.py` | 3.1.2 | Traffic signatures confirming class 3 = Mirai, 4 = Recon, 5 = Web |
| 3 | `scripts/03_leakage_audit.py` | 4.1 | Protocols P0 to P4 (deduplication, IAT removal, capture-time-blocked CV) |
| 4 | `scripts/04_detection_models.py <model>` | 3.5, 4.2, 4.3 | Seven classifiers under P4 with nested randomised tuning; noise robustness |
| 5 | `scripts/05_phase2_stages.py` | 3.3, 4.5 | Merged stages, hierarchical cascade, held-out subtype, leave-one-family-out, cost-sensitive decisions |
| 6 | `scripts/06_shap.py` | 4.6 | TreeSHAP on the leakage-controlled LightGBM |
| 7 | `scripts/07_statistics.py` | 4.2, 4.3 | Main table, 95% CIs, Friedman, Nadeau-Bengio and McNemar tests, prior-shift analysis |
| E | `external/` | 4.4 | Capture-file-disjoint evaluation on the original CICIoT2023 release |
| F | `figures/` | all | Figure 1 and the result figures |

Primary protocol (P4): exact duplicates removed (1,097,192 unique records), IAT excluded (15 features),
5-fold cross-validation in which each class is split into five contiguous capture-time blocks.

## Running

```bash
pip install -r requirements.txt
# put df_FL_CICIoT2023_6_classes.csv in data/ (see data/README.md)
bash run_all.sh
```

Steps 3 and 4 take several hours on a 4-core machine. Steps 6 and 7 and the figure scripts need the intermediate files
(`results/*.npy`, `results/model_*_fold0.pkl`) written by step 4.

## Results included

`results/` holds the JSON outputs used in the paper, so tables and figures can be checked without re-running the models.
`results/table_main.csv` is the main comparison table (P4). `results/external/e1_external.json` holds the external evaluation.

## Data

Datasets are not redistributed here. See `data/README.md` for sources and SHA-256 checksums.

## Licence

Code: MIT (see `LICENSE`). The CICIoT2023 data remain subject to the licence of the Canadian Institute for Cybersecurity.
