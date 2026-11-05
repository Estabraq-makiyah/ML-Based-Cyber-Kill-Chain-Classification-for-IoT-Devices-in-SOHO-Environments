"""Traffic signatures used to verify the class encoding (class 3 = Mirai, class 4 = Recon, class 5 = Web).

Mirai floods in CICIoT2023 are GRE-encapsulated ('greeth', 'greip') or UDP-plain, so class 3 should be dominated by
Protocol Type values above UDP (17), with large, uniform packets. Reconnaissance should show small packets and
SYN/RST activity. Output: results/label_evidence.json (statistics computed on the deduplicated data).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from common import *

dd = pd.read_parquet(os.path.join(DATA, 'dedup.parquet'))
out = {}
for c, s in dd.groupby('sub_label'):
    pt = s['Protocol Type']
    out[str(c)] = {
        'gre_dominated': float((pt > 17).mean()),            # window-averaged protocol number above UDP, i.e. GRE (47) present
        'udp_exact': float((pt == 17).mean()),
        'icmp_mix': float(((pt > 0) & (pt < 6)).mean()),       # averages below TCP (6) imply ICMP (1) in the window
        'tot_size_med': float(s['Tot size'].median()),
        'tot_size_q25': float(s['Tot size'].quantile(.25)),
        'tot_size_q75': float(s['Tot size'].quantile(.75)),
        'min_le60': float((s['Min'] <= 60).mean()),
        'max_ge1000': float((s['Max'] >= 1000).mean()),
        'syn_pos': float((s['syn_count'] > 0).mean()),
        'rst_pos': float((s['rst_count'] > 0).mean()),
        'var0': float((s['Variance'] == 0).mean()),
    }
save('label_evidence', out)
print(pd.DataFrame(out).T.round(3))
