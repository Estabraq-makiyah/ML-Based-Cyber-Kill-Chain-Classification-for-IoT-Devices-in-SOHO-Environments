"""Convert the six-class CICIoT2023 CSV to parquet and build the deduplicated, capture-time-blocked file.

Usage: python scripts/00_prepare_data.py [path/to/df_FL_CICIoT2023_6_classes.csv]
Outputs: data/data.parquet (all 1,800,000 rows) and data/dedup.parquet (1,097,192 unique rows + 'block' column).
"""
import sys, os, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from common import *

src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, 'df_FL_CICIoT2023_6_classes.csv')
h = hashlib.sha256(open(src, 'rb').read()).hexdigest()
print('SHA-256 of input CSV:', h)
df = pd.read_csv(src)
assert list(df.columns) == FEATS + ['sub_label'], 'unexpected columns'
df.to_parquet(os.path.join(DATA, 'data.parquet'))
dd = df.drop_duplicates().reset_index(drop=True)
# Capture-time blocks: within each class, rank records by IAT and cut into five contiguous blocks.
# Fold k of the blocked CV tests on block k of every class, so training and test come from disjoint capture-time ranges.
dd['block'] = dd.groupby('sub_label')['IAT'].rank(method='first', pct=True).mul(5).clip(upper=4.999).astype(int)
dd.to_parquet(os.path.join(DATA, 'dedup.parquet'))
print(f'rows {len(df):,} | unique {len(dd):,} | exact duplicates {len(df)-len(dd):,} ({1-len(dd)/len(df):.1%})')
print(dd.groupby('sub_label').size().rename('unique rows per class'))
