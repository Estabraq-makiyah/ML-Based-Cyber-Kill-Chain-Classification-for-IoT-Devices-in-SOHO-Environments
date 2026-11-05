import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from common import RES, DATA, FIG
import json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#8a8984','axes.labelcolor':'#52514e','xtick.color':'#52514e','ytick.color':'#52514e'})
S=['#2a78d6','#eb6834','#1baf7a']; GREY='#c9c8c2'; INK='#0b0b0b'
d=json.load(open(f'{RES}/s2_leakage.json')); s1=json.load(open(f'{RES}/s1_reproduce.json'))
# Fig R1 duplicates per class
u=d['info']['unique_per_class']; cls=['0 Benign','1 DDoS','2 DoS','3 Mirai','4 Reconnaissance','5 Web attack']
uniq=np.array([u[str(i)] for i in range(6)]); dup=300000-uniq
fig,ax=plt.subplots(figsize=(6.3,2.8)); yy=np.arange(6)[::-1]
ax.barh(yy,uniq/1000,color=S[0],height=0.6,label='Unique rows',edgecolor='white',linewidth=1)
ax.barh(yy,dup/1000,left=uniq/1000,color=GREY,height=0.6,label='Exact duplicates',edgecolor='white',linewidth=1)
for i,(a,b) in enumerate(zip(uniq,dup)): ax.text(301,yy[i],f'{b/3000:.0f}% dup',va='center',fontsize=8,color='#52514e')
ax.set_yticks(yy); ax.set_yticklabels(cls); ax.set_xlabel('Rows (thousands)'); ax.set_xlim(0,340)
ax.legend(frameon=False,loc='lower center',bbox_to_anchor=(0.45,1.0),ncol=2); fig.tight_layout(); fig.savefig(f'{FIG}/FigR1_duplicates.png',dpi=300)
# Fig R2 macro-F1 under stricter protocols
P=[('Original\nrandom split',None),('Deduplicated\nrandom','dedup_random_all'),('Dedup. random\nno IAT','dedup_random_noIAT'),('Dedup. time-blocked\nwith IAT','dedup_blocked_all'),('Dedup. time-blocked\nno IAT','dedup_blocked_noIAT')]
M=['RF','DT','LGBM']; fig,ax=plt.subplots(figsize=(6.8,3.2)); w=0.26; x=np.arange(len(P))
for j,m in enumerate(M):
    vals=[];errs=[]
    for lab,key in P:
        if key is None: vals.append(s1[m]['macroF1']); errs.append(0)
        else:
            r=d['res'][f'{key}|{m}']; vals.append(r.get('macroF1_mean',r.get('macroF1'))); errs.append(r.get('macroF1_sd',0))
    ax.bar(x+(j-1)*w,vals,w-0.03,color=S[j],label={'RF':'Random Forest','DT':'Decision Tree','LGBM':'LightGBM'}[m],yerr=errs,error_kw={'elinewidth':1,'ecolor':'#52514e','capsize':2})
    if m=='RF':
        for xi,v in zip(x,vals): ax.text(xi-w,0.515,f'{v:.2f}',ha='center',fontsize=7,color='white',fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels([p[0] for p in P],fontsize=7.5); ax.set_ylim(0.5,1.05); ax.set_ylabel('Macro-F1 (test / mean ± SD over 5 folds)')
ax.yaxis.grid(True,color='#e6e5e0'); ax.set_axisbelow(True); ax.legend(frameon=False,ncol=3,loc='lower center',bbox_to_anchor=(0.5,1.0))
fig.tight_layout(); fig.savefig(f'{FIG}/FigR2_protocols.png',dpi=300); print('ok')
