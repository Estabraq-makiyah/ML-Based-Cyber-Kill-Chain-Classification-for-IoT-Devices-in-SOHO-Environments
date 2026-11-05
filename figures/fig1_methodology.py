import os, matplotlib; matplotlib.use('Agg')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output'); os.makedirs(OUT, exist_ok=True)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family': 'DejaVu Sans'})
INK = '#1f2328'; MUT = '#4a4946'
BLUE_F = '#e8f0fb'; BLUE_E = '#2a78d6'; ORG_F = '#fdeee6'; ORG_E = '#d9622b'
GRY_F = '#f3f3f1'; GRY_E = '#8a8984'; GRN_F = '#e6f5ee'; GRN_E = '#1a9e6e'
fig, ax = plt.subplots(figsize=(11.5, 9.0)); ax.set_xlim(0, 100); ax.set_ylim(0, 90); ax.axis('off')

def rbox(x, y, w, h, fc, ec, lw=1.3, r=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw))
def arrow(p, q, conn='arc3', color=MUT, lw=1.6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=15, color=color, lw=lw, connectionstyle=conn, shrinkA=0, shrinkB=0))
def topbox(x, title, lines, fc, ec):
    rbox(x, 73, 30, 15, fc, ec)
    ax.text(x + 15, 85.0, title, ha='center', va='center', fontsize=14, fontweight='bold', color=INK)
    for i, l in enumerate(lines):
        ax.text(x + 15, 81.0 - 3.2 * i, l, ha='center', va='center', fontsize=11.5, color=MUT)

topbox(1, 'Dataset', ['CICIoT2023 six-class release', '1,800,000 records', '16 features, 6 classes'], GRY_F, GRY_E)
topbox(35, 'Data audit', ['Exact duplicates (39.0%)', 'Label verification', 'IAT capture-time shortcut'], BLUE_F, BLUE_E)
topbox(69, 'Leakage controls', ['1,097,192 unique records', 'IAT excluded (15 features)', 'Capture-time-blocked 5-fold CV'], BLUE_F, BLUE_E)
arrow((31, 80.5), (35, 80.5)); arrow((65, 80.5), (69, 80.5))

def phase(x, title, sub, items, fc, ec):
    rbox(x, 17, 45, 52, 'white', ec, lw=1.8, r=1.8)
    ax.text(x + 22.5, 65.4, title, ha='center', va='center', fontsize=15, fontweight='bold', color=ec)
    ax.text(x + 22.5, 61.9, sub, ha='center', va='center', fontsize=11.2, color=MUT, style='italic')
    for i, (t, d) in enumerate(items):
        yy = 51.6 - i * 8.2
        rbox(x + 2, yy, 41, 7.2, fc, ec, lw=1.0, r=0.9)
        ax.text(x + 4, yy + 4.85, t, ha='left', va='center', fontsize=12, fontweight='bold', color=INK)
        ax.text(x + 4, yy + 2.0, d, ha='left', va='center', fontsize=10.5, color=MUT)

phase(1, 'Phase 1: intrusion detection', 'six classes: benign and five attack categories', [
    ('Seven classifiers', 'Dummy, LR, MLP, DT, RF, LightGBM, XGBoost'),
    ('Nested tuning', '15 random draws, inner blocked 3-fold CV'),
    ('Leakage audit', 'protocols P0 to P4'),
    ('Robustness', 'feature noise, class-prior shift'),
    ('External replication', 'independent release, capture-file-disjoint CV')], BLUE_F, BLUE_E)
phase(54, 'Phase 2: stage-level output', 'Cyber Kill Chain stages (S1, S3, S4, S6)', [
    ('Merged stages', 'Delivery = DDoS + DoS'),
    ('Hierarchical cascade', 'binary detector, then stage classifier'),
    ('Held-out subtype', 'unseen DoS or DDoS assigned to Delivery'),
    ('Leave-one-family-out', 'detection of unseen attack families'),
    ('Cost-sensitive decisions', 'minimum expected cost')], ORG_F, ORG_E)

# leakage controls -> both phases
ax.plot([84, 84], [73, 71.2], color=MUT, lw=1.6, solid_capstyle='butt')
ax.plot([23.5, 84], [71.2, 71.2], color=MUT, lw=1.6, solid_capstyle='butt')
arrow((23.5, 71.2), (23.5, 69)); arrow((76.5, 71.2), (76.5, 69))
# phase 1 -> phase 2
arrow((46, 43), (54, 43), color=ORG_E)
ax.text(50, 44.0, 'LightGBM\nsettings', ha='center', va='bottom', fontsize=9, color=ORG_E, linespacing=0.95)

# SHAP
rbox(28, 2, 44, 10.5, GRN_F, GRN_E)
ax.text(50, 9.2, 'SHAP explainability', ha='center', va='center', fontsize=14, fontweight='bold', color=INK)
ax.text(50, 5.3, 'TreeSHAP on the leakage-controlled LightGBM', ha='center', va='center', fontsize=11, color=MUT)
arrow((23.5, 17), (28, 7.2), conn='angle,angleA=-90,angleB=0,rad=0')
arrow((76.5, 17), (72, 7.2), conn='angle,angleA=-90,angleB=180,rad=0')
ax.text(1.5, 7.2, 'Metrics: macro-F1 (95% CI),\nbalanced accuracy, MCC;\nFriedman, Nadeau-Bengio\nand McNemar tests', ha='left', va='center', fontsize=9.3, color=MUT, linespacing=1.2)

fig.savefig(os.path.join(OUT,'Methodology_revised.png'), dpi=300, bbox_inches='tight', pad_inches=0.05)
fig.savefig(os.path.join(OUT,'Methodology_revised.pdf'), bbox_inches='tight', pad_inches=0.05)
print('ok')
