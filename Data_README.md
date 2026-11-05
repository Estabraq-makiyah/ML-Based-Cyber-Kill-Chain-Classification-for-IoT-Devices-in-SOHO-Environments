# Data

No data files are stored in this repository. Two public sources are used.

## 1. Six-class CICIoT2023 release (main experiments)

File: `df_FL_CICIoT2023_6_classes.csv` (1,800,000 records, 16 features + `sub_label`, 300,000 records per class).
Place it in this folder, then run `python scripts/00_prepare_data.py`.

SHA-256 of the file used for the paper:
`9183602851151c803820a248f3c13107b7969c045e5a2cfdfb98aae5d97b2d00`

Verified label encoding: 0 Benign, 1 DDoS, 2 DoS, 3 Mirai, 4 Reconnaissance, 5 Web-based attack
(see `scripts/02_label_verification.py`).

## 2. Original CICIoT2023 release (external evaluation)

The original CICIoT2023 dataset (Neto et al., 2023, Canadian Institute for Cybersecurity, University of New Brunswick),
cleaned into a single parquet file of 46,776,697 records with a `source_file` column naming the capture file of each
record. See `external/README.md`.

SHA-256 of the cleaned parquet used for the paper:
`788b23fae99ab59cbe838add0dfc5d8b114fc6fad512855a7348ea1fbafaf1d0`
