"""Figure 3 (feature correlation heatmap) and Figure 11 (literature comparison)."""
import json, os, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from common import FEATS, load, RES, FIG
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#8a8984', 'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e'})
OUT = sys.argv[1] if len(sys.argv) > 1 else FIG
os.makedirs(OUT, exist_ok=True)

# 1. Correlation heatmap: 16 features, Pearson, ordered by average-linkage clustering on 1-|r|
X = load()[FEATS]
X.describe().T[['mean', 'std', 'min', '50%', 'max']].to_csv(os.path.join(RES, 'feature_statistics.csv'))   # Table 8
C = X.corr()
order = leaves_list(linkage(squareform(1 - C.abs().values, checks=False), 'average'))
Co = C.iloc[order, order]
fig, ax = plt.subplots(figsize=(7.2, 6.2))
im = ax.imshow(Co.values, cmap='RdBu_r', vmin=-1, vmax=1)
for i in range(16):
    for j in range(16):
        v = Co.values[i, j]
        ax.text(j, i, f'{(0.0 if abs(v) < 0.005 else v):.2f}', ha='center', va='center', fontsize=6.3, color='white' if abs(v) > 0.6 else '#1f2328')
ax.set_xticks(range(16)); ax.set_yticks(range(16))
ax.set_xticklabels(Co.columns, rotation=45, ha='right'); ax.set_yticklabels(Co.columns)
ax.spines[:].set_visible(False); ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label('Pearson r'); cb.outline.set_visible(False)
fig.tight_layout(); fig.savefig(f'{OUT}/correlation_heatmap.png', dpi=300)

# 2. Literature comparison: reported accuracy vs this study under both protocols
s1 = json.load(open(os.path.join(RES, 's1_reproduce.json')))
p4 = json.load(open(os.path.join(RES, 's3_XGB.json')))
p4_acc = float(np.mean([f['acc'] for f in p4['folds']]))
lit = [  # (label, dataset group, reported accuracy %) as cited in Sections 2 and 5.4
    ('Kumar et al. (2024), DT/RF', 'CICIoT2023', 100.00),
    ('Hajjouz and Avksentieva (2024), CatBoost', 'CICIoT2023', 99.40),
    ('Gheni and Al-Yaseen (2024), MLP', 'CICIoT2023', 96.26),
    ('Soliman et al. (2023), Bi-LSTM', 'ToN-IoT', 99.99),
    ('Balega et al. (2024), XGBoost', 'IoT-23 / NSL-KDD', 99.98),
    ('Rashid et al. (2020), stacking', 'CICIDS2017', 99.90),
    ('El-Sofany et al. (2024), RF-BPNN', 'Custom SDN/NFV', 99.90),
    ('Mahdi (2024), LSTM', 'UNSW-NB15', 98.00),
    ('M et al. (2024), RF', 'UNSW-NB15', 92.73),
    ('Ahmed et al. (2021), Naive Bayes', 'APT alerts (CKC stages)', 91.10),
]
rows = [('This study, original protocol (random split), RF', 'CICIoT2023 six-class', 100 * s1['RF']['acc'], 'p0'),
        ('This study, leakage-controlled (P4), XGBoost', 'CICIoT2023 six-class', 100 * p4_acc, 'p4')]
rows += [(a, b, c, 'lit') for a, b, c in lit]
col = {'p0': '#eb6834', 'p4': '#2a78d6', 'lit': '#c9c8c2'}
fig, ax = plt.subplots(figsize=(6.2, 5.0))
y = np.arange(len(rows))[::-1]
for yi, (lab, ds, acc, k) in zip(y, rows):
    ax.barh(yi, acc, color=col[k], height=0.66)
    ax.text(acc + 0.8, yi, f'{acc:.2f}%', va='center', fontsize=8.2, color='#1f2328')
ax.set_yticks(y); ax.set_yticklabels([f'{lab}\n[{ds}]' for lab, ds, _, _ in rows], fontsize=8.2, linespacing=0.95)
ax.set_xlim(0, 112); ax.set_xticks([0, 20, 40, 60, 80, 100]); ax.set_xlabel('Reported accuracy (%)')
ax.xaxis.grid(True, color='#e6e5e0'); ax.set_axisbelow(True)
ax.axhline(y[1] - 0.5, color='#8a8984', lw=0.8, ls='--')
fig.tight_layout(); fig.savefig(f'{OUT}/literature_comparison.png', dpi=300)
print('P4 accuracy XGB', round(p4_acc, 4), 'P0 RF', round(s1['RF']['acc'], 4))
