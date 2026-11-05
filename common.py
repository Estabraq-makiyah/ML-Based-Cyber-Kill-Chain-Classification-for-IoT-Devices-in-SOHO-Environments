"""Shared paths, feature lists and helpers. All paths are relative to the repository root."""
import os, json, time
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'figures', 'output')
for d in (DATA, RES, FIG):
    os.makedirs(d, exist_ok=True)

FEATS = ['flow_duration', 'Header_Length', 'Protocol Type', 'Rate', 'Srate', 'syn_count', 'urg_count', 'rst_count',
         'Tot sum', 'Min', 'Max', 'AVG', 'Tot size', 'IAT', 'Magnitue', 'Variance']
FEATS_NOIAT = [f for f in FEATS if f != 'IAT']
CLASSES = {0: 'Benign', 1: 'DDoS', 2: 'DoS', 3: 'Mirai', 4: 'Recon', 5: 'Web'}   # verified encoding (see 02_label_verification.py)


def load():
    """Six-class release as distributed (1,800,000 rows, duplicates included)."""
    return pd.read_parquet(os.path.join(DATA, 'data.parquet'))


def save(name, obj):
    json.dump(obj, open(os.path.join(RES, f'{name}.json'), 'w'), indent=1, default=float)
