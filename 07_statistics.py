import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from common import *
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
dd=pd.read_parquet(os.path.join(DATA,'dedup.parquet')); y=dd.sub_label.values; B=dd.block.values
NAMES={'Dummy':'Stratified dummy','LR':'Logistic regression','MLP':'MLP (128-64)','DT':'Decision Tree (tuned)','RF':'Random Forest (tuned)','LGBM':'LightGBM (tuned)','XGB':'XGBoost (tuned)'}
avail=[m for m in NAMES if os.path.exists(f'{RES}/s3_{m}.json') and 'oof' in json.load(open(f'{RES}/s3_{m}.json'))]
R={m:json.load(open(f'{RES}/s3_{m}.json')) for m in avail}
rows=[]
for m in avail:
    f=R[m]['folds']; mf=np.array([x['macroF1'] for x in f]); k=len(mf)
    ci=stats.t.ppf(.975,k-1)*mf.std(ddof=1)/np.sqrt(k)
    pc=np.mean([x['f1_per_class'] for x in f],0)
    rows.append({'model':NAMES[m],'key':m,'macroF1_mean':mf.mean(),'macroF1_sd':mf.std(ddof=1),'ci95':ci,'bal_acc':np.mean([x['bal_acc'] for x in f]),
      'mcc':np.mean([x['mcc'] for x in f]),'acc':np.mean([x['acc'] for x in f]),**{f'F1_c{i}':pc[i] for i in range(6)},
      'train_s':np.mean([x['train_s'] for x in f]),'infer_ms_per_1k':np.mean([x['infer_s_per_1k'] for x in f])*1000,
      'model_MB':np.mean([x.get('model_MB',np.nan) for x in f]),'oof_macroF1':R[m]['oof']['macroF1']})
T=pd.DataFrame(rows).sort_values('macroF1_mean',ascending=False); T.to_csv(os.path.join(RES,'table_main.csv'),index=False)
out={'table':T.to_dict('records')}
# Friedman across folds (tuned/learned models only)
cand=[m for m in avail if m!='Dummy']
M=np.array([[x['macroF1'] for x in R[m]['folds']] for m in cand])
if len(cand)>=3: out['friedman']=dict(zip(['stat','p'],stats.friedmanchisquare(*M)))
# Nadeau-Bengio corrected resampled t-test: best vs others
best=T.key.iloc[0]; n_te=len(y)/5; n_tr=len(y)-n_te
def nb(a,b):
    d=np.array(a)-np.array(b); k=len(d); v=d.var(ddof=1)
    t=d.mean()/np.sqrt((1/k+n_te/n_tr)*v) if v>0 else np.inf; return {'mean_diff':d.mean(),'t':t,'p':2*stats.t.sf(abs(t),k-1)}
out['nb_vs_best']={m:nb([x['macroF1'] for x in R[best]['folds']],[x['macroF1'] for x in R[m]['folds']]) for m in cand if m!=best}
# McNemar on pooled OOF predictions: best vs each
pb=np.load(f'{RES}/oof_{best}.npy').argmax(1)==y
out['mcnemar_vs_best']={}
for m in cand:
    if m==best: continue
    po=np.load(f'{RES}/oof_{m}.npy').argmax(1)==y
    tab=[[np.sum(pb&po),np.sum(pb&~po)],[np.sum(~pb&po),np.sum(~pb&~po)]]
    r=mcnemar(tab,exact=False,correction=True); out['mcnemar_vs_best'][m]={'b_best_only':int(tab[0][1]),'c_other_only':int(tab[1][0]),'stat':r.statistic,'p':r.pvalue}
# Noise robustness
out['noise']={m:{s:np.mean([x['noise'][s] for x in R[m]['folds']]) for s in R[m]['folds'][0]['noise']} for m in avail if 'noise' in R[m]['folds'][0]}
# Prior-shift: binary attack decision from OOF (pred != benign)
out['prior_shift']={}
for m in cand:
    p=np.load(f'{RES}/oof_{m}.npy').argmax(1); att=y>0
    TPR=float((p[att]>0).mean()); FPR=float((p[~att]>0).mean()); res={'TPR':TPR,'FPR':FPR}
    for pi in [0.5,0.1,0.01,0.001]:
        prec=TPR*pi/(TPR*pi+FPR*(1-pi)); res[str(pi)]={'precision':prec,'false_alerts_per_1M':FPR*(1-pi)*1e6,'true_alerts_per_1M':TPR*pi*1e6}
    out['prior_shift'][m]=res
save('s6_stats',out)
pd.set_option('display.width',250); print(T[['model','macroF1_mean','macroF1_sd','ci95','bal_acc','mcc','train_s','model_MB']].round(4))
print(json.dumps({k:out[k] for k in out if k!='table'},default=float,indent=0)[:3000])
