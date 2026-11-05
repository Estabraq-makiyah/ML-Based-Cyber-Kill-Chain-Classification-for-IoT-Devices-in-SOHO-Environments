import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from common import RES, DATA, FIG
import json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#8a8984','axes.labelcolor':'#52514e','xtick.color':'#52514e','ytick.color':'#52514e'})
S=['#2a78d6','#eb6834','#1baf7a','#eda100']; INK='#0b0b0b'; MUT='#52514e'; GRID='#e6e5e0'
d=json.load(open(f'{RES}/s6_stats.json')); T=pd.DataFrame(d['table'])
# R3: model comparison, mean macro-F1 with 95% CI and fold points
fig,ax=plt.subplots(figsize=(6.3,3.0)); T=T.iloc[::-1].reset_index(drop=True)
for i,r in T.iterrows():
    f=[x['macroF1'] for x in json.load(open(f"results/s3_{r.key}.json"))['folds']]
    ax.scatter(f,[i]*5,s=14,color='#c9c8c2',zorder=2)
    ax.errorbar(r.macroF1_mean,i,xerr=r.ci95,fmt='o',color=S[0],ms=6,elinewidth=1.5,capsize=3,zorder=3)
    ax.text(0.98,i,f'{r.macroF1_mean:.3f} ± {r.ci95:.3f}',va='center',ha='right',fontsize=8,color=INK,transform=ax.get_yaxis_transform())
ax.set_yticks(range(len(T))); ax.set_yticklabels(T.model); ax.set_xlim(0.1,1.05); ax.set_xlabel('Macro-F1 under P4 (dot = mean, bar = 95% CI, grey = individual folds)')
ax.xaxis.grid(True,color=GRID); ax.set_axisbelow(True); fig.tight_layout(); fig.savefig(f'{FIG}/FigR3_models.png',dpi=300)
# R4: noise robustness
N=d['noise']; names={'DT':'Decision Tree','RF':'Random Forest','LGBM':'LightGBM','XGB':'XGBoost'}
T2=pd.DataFrame(d['table']).set_index('key')
fig,ax=plt.subplots(figsize=(5.2,3.0)); xs=[0,0.05,0.1,0.2,0.5]
for j,m in enumerate(['RF','LGBM','XGB','DT']):
    ys=[T2.loc[m,'macroF1_mean']]+[N[m][str(s)] for s in xs[1:]]
    ax.plot(range(5),ys,color=S[j],lw=2,marker='o',ms=5,label=names[m])
    ax.text(4.08,ys[-1],names[m],va='center',fontsize=8,color=MUT)
ax.set_xticks(range(5)); ax.set_xticklabels(['clean','σ=0.05','σ=0.1','σ=0.2','σ=0.5']); ax.set_xlim(-0.2,5.0)
ax.set_ylabel('Macro-F1'); ax.set_xlabel('Multiplicative log-normal noise on continuous features'); ax.yaxis.grid(True,color=GRID)
ax.legend(frameon=False,ncol=4,loc='lower center',bbox_to_anchor=(0.45,1.0),fontsize=8); fig.tight_layout(); fig.savefig(f'{FIG}/FigR4_noise.png',dpi=300)
# R5: prior shift precision
P=d['prior_shift']; pis=[0.5,0.1,0.01,0.001]
fig,ax=plt.subplots(figsize=(5.2,3.0))
for j,m in enumerate(['XGB','LGBM','RF','DT']):
    ys=[P[m][str(p)]['precision'] for p in pis]
    ax.plot(range(4),ys,color=S[j],lw=2,marker='o',ms=5,label=f"{names[m]} (FPR {P[m]['FPR']:.1%})")
ax.set_xticks(range(4)); ax.set_xticklabels(['50%\n(test prior)','10%','1%','0.1%']); ax.set_xlabel('Attack prevalence in traffic'); ax.set_ylabel('Precision of attack alerts')
ax.set_ylim(0,1); ax.yaxis.grid(True,color=GRID); ax.legend(frameon=False,fontsize=7.5,loc='upper right'); fig.tight_layout(); fig.savefig(f'{FIG}/FigR5_priorshift.png',dpi=300)
# R6: normalised confusion matrix, XGB OOF
dd=pd.read_parquet(os.path.join(DATA,'dedup.parquet')); y=dd.sub_label.values; p=np.load(f'{RES}/oof_XGB.npy').argmax(1)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y,p,normalize='true')
lab=['0 Benign','1 DDoS','2 DoS','3 Mirai','4 Recon','5 Web']
fig,ax=plt.subplots(figsize=(4.6,3.9)); im=ax.imshow(cm,cmap='Blues',vmin=0,vmax=1)
for i in range(6):
    for j in range(6): ax.text(j,i,f'{cm[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if cm[i,j]>0.55 else INK)
ax.set_xticks(range(6)); ax.set_yticks(range(6)); ax.set_xticklabels(lab,rotation=30,ha='right'); ax.set_yticklabels(lab)
ax.set_xlabel('Predicted'); ax.set_ylabel('True (row-normalised recall)'); ax.spines[:].set_visible(False)
fig.colorbar(im,ax=ax,shrink=0.8); fig.tight_layout(); fig.savefig(f'{FIG}/FigR6_cm_xgb.png',dpi=300); print('ok')
