# External evaluation

An independent check on the original CICIoT2023 release, using capture-file-disjoint cross-validation
(no capture file contributes records to both training and test sets).

1. `build_external_sample.py path/to/ciciot2023.parquet` rebuilds the exact 747,267-record sample (305 capture files)
   from the cleaned full release, using the row numbers in `external_sample_index.csv.gz`.
   The script checks that the input has 46,776,697 rows.
2. `external_evaluation.py` uses the 39 numeric columns of the release as model inputs (38 traffic features and the
   `rate_was_infinite` flag); labels, the capture-file name and any row-position column are never used as inputs.
   It trains LightGBM (four feature sets: all, without IAT, without TTL, without both),
   XGBoost and Random Forest, and compares a random record-level split with 4 capture-file-disjoint folds.
   Output: `results/external/e1_external.json`.

The six categories (Benign, DDoS, DoS, Mirai, Recon, Web) come from the `category` column of the cleaned release.
3. `external_stage_merge.py` collapses the confusion matrices to the kill-chain stages of Table 6 (DDoS and DoS
   merged into Delivery) and writes the stage-level scores and the confusion rates quoted in Sections 4.4 and 4.5
   to `results/external/e1_stage_merge.json`.

Note on an earlier version: a previous run of `external_evaluation.py` also used `file_row_number`, the position of a
record within its capture file, as an input. That column identifies the capture and the position in it rather than
traffic behaviour, and it inflated the random-split scores (macro-F1 0.992 to 0.998). It is now excluded, and all
external results in the paper come from the corrected run.
