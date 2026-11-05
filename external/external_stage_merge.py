"""Stage-level view of the external evaluation (Section 4.5).

Collapses the six-category confusion matrices in results/external/e1_external.json to the
five kill-chain stages of Table 6 (DDoS and DoS merged into Delivery) and reports, per model,
the Delivery F1 and stage-level macro-F1 for every file-disjoint fold, plus the main
cross-category confusion rates quoted in Section 4.4.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from common import RES

CATS = ['Benign', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Web']
STAGE = [0, 1, 1, 2, 3, 4]  # Benign, Delivery, C2 (Mirai), Reconnaissance, Exploitation (Web)


def collapse(cm):
    cm = np.asarray(cm, dtype=float)
    s = np.zeros((5, 5))
    for i in range(6):
        for j in range(6):
            s[STAGE[i], STAGE[j]] += cm[i, j]
    return s


def f1(cm):
    tp = np.diag(cm)
    return 2 * tp / (cm.sum(0) + cm.sum(1))


def rates(cm):
    r = np.asarray(cm, dtype=float)
    r = r / r.sum(1, keepdims=True)
    return {'DDoS_to_DoS': r[1, 2], 'DoS_to_DDoS': r[2, 1], 'Recon_to_Benign': r[4, 0],
            'Benign_to_Recon': r[0, 4], 'Web_to_Benign': r[5, 0], 'Web_to_Recon': r[5, 4]}


d = json.load(open(os.path.join(RES, 'external', 'e1_external.json')))
out = {}
for key, v in d.items():
    if '|' not in key:
        continue
    folds = [collapse(f['cm']) for f in v['file_disjoint_folds']]
    out[key] = {
        'stage_macroF1_folds': [float(f1(s).mean()) for s in folds],
        'stage_macroF1_mean': float(np.mean([f1(s).mean() for s in folds])),
        'category_macroF1_mean': v['file_disjoint_macroF1_mean'],
        'delivery_F1_folds': [float(f1(s)[1]) for s in folds],
        'confusion_rates_pooled': {k: float(x) for k, x in rates(v['file_disjoint_pooled']['cm']).items()},
        'confusion_rates_random': {k: float(x) for k, x in rates(v['random_split']['cm']).items()},
    }
    print(key, 'stage macro-F1 %.3f (category %.3f)' % (out[key]['stage_macroF1_mean'], out[key]['category_macroF1_mean']),
          'Delivery F1 min %.3f' % min(out[key]['delivery_F1_folds']))
json.dump(out, open(os.path.join(RES, 'external', 'e1_stage_merge.json'), 'w'), indent=1)
