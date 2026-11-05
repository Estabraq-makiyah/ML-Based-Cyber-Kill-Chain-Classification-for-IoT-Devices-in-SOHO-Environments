import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from common import RES, DATA, FIG
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#8a8984','axes.labelcolor':'#52514e','xtick.color':'#52514e','ytick.color':'#52514e'})
d=json.load(open(f'{RES}/s5_shap.json')); F=np.array(d['features']); pcl=np.array(d['per_class_mean_abs_shap'])/6  # contribution to mean over classes
order=np.argsort(pcl.sum(0))[::-1][:10][::-1]
cols=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300']; labs=['Benign','DDoS','DoS','Mirai','Recon','Web']
fig,ax=plt.subplots(figsize=(5.6,3.4)); left=np.zeros(len(order))
for c in range(6):
    ax.barh(range(len(order)),pcl[c,order],left=left,color=cols[c],height=0.62,label=labs[c],edgecolor='white',linewidth=1); left+=pcl[c,order]
ax.set_yticks(range(len(order))); ax.set_yticklabels(F[order]); ax.set_xlabel('Mean |SHAP| (log-odds), averaged over classes')
ax.legend(frameon=False,ncol=6,fontsize=7,loc='lower center',bbox_to_anchor=(0.4,1.0),handlelength=1); ax.xaxis.grid(True,color='#e6e5e0'); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(f'{FIG}/FigR7_shap.png',dpi=300); print('ok')
