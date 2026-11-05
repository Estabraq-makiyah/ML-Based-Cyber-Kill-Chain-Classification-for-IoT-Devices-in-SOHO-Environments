# Leakage diagnostics: dedup, IAT shortcut, pseudo-session (IAT-blocked) split
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from common import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
import lightgbm as lgb, sys
df=load()
n0=len(df); dd=df.drop_duplicates().reset_index(drop=True)
# near-duplicates: identical after rounding to 3 significant figures
Xr=dd[FEATS].to_numpy()
with np.errstate(divide='ignore'):
    mag=np.where(Xr!=0,np.floor(np.log10(np.abs(Xr))),0)
Xr=np.round(Xr/10**(mag-2))*10**(mag-2)
nd=pd.DataFrame(Xr).duplicated().sum()
info={'n_rows':n0,'n_unique':len(dd),'n_exact_dup':n0-len(dd),'unique_per_class':dd.sub_label.value_counts().sort_index().to_dict(),
      'near_dup_3sf_among_unique':int(nd)}
# blocked folds: per class, rank by IAT, 5 contiguous blocks
dd['block']=dd.groupby('sub_label')['IAT'].rank(method='first',pct=True).mul(5).clip(upper=4.999).astype(int)

y=dd.sub_label.values
def mk(n):
    return {'RF':RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=2),
            'DT':DecisionTreeClassifier(random_state=42),
            'LGBM':lgb.LGBMClassifier(n_estimators=100,learning_rate=0.1,random_state=42,verbose=-1)}[n]
def ev(yt,p): return {'acc':accuracy_score(yt,p),'bal_acc':balanced_accuracy_score(yt,p),'macroF1':f1_score(yt,p,average='macro'),
    'f1_per_class':f1_score(yt,p,average=None).tolist()}
import gc,os
os.makedirs(f'{RES}/s2_parts',exist_ok=True)
res={}
def cached(key,fn):
    fp=f'{RES}/s2_parts/{key.replace("|","__")}.json'
    if os.path.exists(fp): return json.load(open(fp))
    r=fn(); json.dump(r,open(fp,'w'),default=float); gc.collect(); return r
XA=dd[FEATS].to_numpy(np.float32); XN=dd[FEATS_NOIAT].to_numpy(np.float32)

# P1/P2: dedup random split
tr,te=train_test_split(np.arange(len(dd)),test_size=.2,stratify=y,random_state=42)
for feats,tag in [(FEATS,'dedup_random_all'),(FEATS_NOIAT,'dedup_random_noIAT')]:
    for n in ['DT','LGBM','RF']:
        XX=XA if feats is FEATS else XN
        def f1(XX=XX,n=n):
            m=mk(n).fit(XX[tr],y[tr]); p=m.predict(XX[te]); del m; return ev(y[te],p)
        res[f'{tag}|{n}']=cached(f'{tag}|{n}',f1); print(tag,n,res[f'{tag}|{n}']['macroF1'],flush=True)
# P3/P4: blocked 5-fold
for feats,tag in [(FEATS,'dedup_blocked_all'),(FEATS_NOIAT,'dedup_blocked_noIAT')]:
    for n in ['DT','LGBM','RF']:
        XX=XA if feats is FEATS else XN
        fs=[]
        for k in range(5):
            trm=dd.block.values!=k
            def f2(XX=XX,n=n,trm=trm):
                m=mk(n).fit(XX[trm],y[trm]); p=m.predict(XX[~trm]); del m; return ev(y[~trm],p)
            fs.append(cached(f'{tag}|{n}|{k}',f2))
        res[f'{tag}|{n}']={'folds':fs,'macroF1_mean':np.mean([f['macroF1'] for f in fs]),'macroF1_sd':np.std([f['macroF1'] for f in fs],ddof=1),
            'bal_acc_mean':np.mean([f['bal_acc'] for f in fs]),'acc_mean':np.mean([f['acc'] for f in fs]),
            'f1_per_class_mean':np.mean([f['f1_per_class'] for f in fs],0).tolist()}
        print(tag,n,res[f'{tag}|{n}']['macroF1_mean'],flush=True)
# IAT-only on blocked vs random
for tag,split in [('IATonly_random',(tr,te))]:
    m=DecisionTreeClassifier(max_depth=12,random_state=0).fit(dd.loc[tr,['IAT']],y[tr]); res['IATonly_dedup_random']=ev(y[te],m.predict(dd.loc[te,['IAT']]))
save('s2_leakage',{'info':info,'res':res})
