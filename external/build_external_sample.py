"""Rebuild the exact external-evaluation sample (747,267 records, 305 capture files) from the cleaned full
CICIoT2023 release (46,776,697 records, one parquet file with a 'source_file' column).

Usage: python external/build_external_sample.py path/to/ciciot2023.parquet
Output: data/ext_sample.parquet
The sample is defined by external_sample_index.csv.gz, which lists the row numbers (0-based, in file order) of the
selected records, so the sample is identical regardless of the sampling code.
"""
import sys, os, time, duckdb
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = sys.argv[1].replace('\\', '/')
idx = os.path.join(ROOT, 'external', 'external_sample_index.csv.gz').replace('\\', '/')
out = os.path.join(ROOT, 'data', 'ext_sample.parquet').replace('\\', '/')
os.makedirs(os.path.dirname(out), exist_ok=True)
con = duckdb.connect(); t = time.time()
n_full = con.execute(f"SELECT count(*) FROM read_parquet('{src}')").fetchone()[0]
assert n_full == 46_776_697, f'expected 46,776,697 rows in the full release, found {n_full:,}'
con.execute(f"""
COPY (
  SELECT f.* EXCLUDE (file_row_number)
  FROM read_parquet('{src}', file_row_number = true) f
  JOIN read_csv('{idx}') i USING (file_row_number)
  ORDER BY file_row_number
) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
n, nf = con.execute(f"SELECT count(*), count(DISTINCT source_file) FROM read_parquet('{out}')").fetchone()
print(f'sample rows {n:,} from {nf} capture files in {time.time()-t:.0f} s -> {out}')
