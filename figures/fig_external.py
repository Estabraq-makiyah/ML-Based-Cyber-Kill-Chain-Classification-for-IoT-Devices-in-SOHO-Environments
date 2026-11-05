import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from common import RES, DATA, FIG
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#8a8984','axes.labelcolor':'#52514e','xtick.color':'#52514e','ytick.color':'#52514e'})
d=json.load(open(os.path.join(RES,'external','e1_external.json'))); r=d['LGBM|noIAT_noTTL']
cats=['Benign','DDoS','DoS','Mirai','Recon','Web']; x=np.arange(6); w=0.38
a=r['random_split']['f1_per_class']; b=r['f1_per_class_mean']
fig,ax=plt.subplots(figsize=(6.0,3.0))
ax.bar(x-w/2,a,w-0.03,color='#c9c8c2',label='Random record-level split')
ax.bar(x+w/2,b,w-0.03,color='#2a78d6',label='Capture-file-disjoint (mean of 4 folds)')
for xi,v in zip(x,b): ax.text(xi+w/2,v+0.02,f'{v:.2f}',ha='center',fontsize=7.5,color='#0b0b0b')
ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_ylim(0,1.12); ax.set_ylabel('F1 (LightGBM)')
ax.yaxis.grid(True,color='#e6e5e0'); ax.set_axisbelow(True); ax.legend(frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(0.5,1.0),fontsize=8)
fig.tight_layout(); fig.savefig(f'{FIG}/FigR8_external.png',dpi=300); print('ok')
