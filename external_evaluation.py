# External evaluation on the independently processed CICIoT2023 release (capture-file-disjoint)
import pandas as pd, numpy as np, json, time, gc, sys, os, warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, matthews_corrcoef, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb, xgboost as xgb
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT=os.path.join(ROOT,'results','external'); os.makedirs(OUT,exist_ok=True)
df=pd.read_parquet(os.path.join(ROOT,'data','ext_sample.parquet'))
CATS=['Benign','DDoS','DoS','Mirai','Recon','Web']; y=df.category.map({c:i for i,c in enumerate(CATS)}).values
# model inputs: traffic features only (identifiers, labels and any row-position columns excluded)
ALL=[c for c in df.columns if c not in ('label','category','binary_label','source_file') and not c.startswith('file_row_number')]
SETS={'all':ALL,'noIAT':[c for c in ALL if c!='inter_arrival_time'],'noTTL':[c for c in ALL if c!='time_to_live'],
      'noIAT_noTTL':[c for c in ALL if c not in ('inter_arrival_time','time_to_live')]}
# file-disjoint folds: within each category, files shuffled (seed 0) and dealt round-robin to 4 folds
rng=np.random.default_rng(0); fold_of={}
for c in CATS:
    files=sorted(df.loc[df.category==c,'source_file'].unique()); rng.shuffle(files)
    for i,f in enumerate(files): fold_of[f]=i%4
G=df.source_file.map(fold_of).values
def mk(n):
    if n=='LGBM': return lgb.LGBMClassifier(n_estimators=800,learning_rate=0.05,num_leaves=127,min_child_samples=10,colsample_bytree=0.6,random_state=0,verbose=-1,n_jobs=2)
    if n=='XGB': return xgb.XGBClassifier(n_estimators=800,learning_rate=0.1,max_depth=10,colsample_bytree=0.8,reg_lambda=1,tree_method='hist',random_state=0,n_jobs=2,verbosity=0)
    if n=='RF': return RandomForestClassifier(n_estimators=100,max_features='sqrt',random_state=0,n_jobs=2)
def ev(a,b): return {'macroF1':f1_score(a,b,average='macro'),'bal_acc':balanced_accuracy_score(a,b),'acc':accuracy_score(a,b),'mcc':matthews_corrcoef(a,b),
                     'f1_per_class':f1_score(a,b,average=None,labels=range(6)).tolist(),'cm':confusion_matrix(a,b,labels=range(6)).tolist()}
out={'n':len(df),'per_cat':df.category.value_counts().to_dict(),'files_per_cat':df.groupby('category').source_file.nunique().to_dict(),
     'fold_test_files':{str(k):sorted([f for f,v in fold_of.items() if v==k]) for k in range(4)}}
jobs=[(m,fs) for fs in SETS for m in ['LGBM']]+[(m,'noIAT_noTTL') for m in ['XGB','RF']]
for m,fs in jobs:
    X=df[SETS[fs]].to_numpy(np.float32)
    tr,te=train_test_split(np.arange(len(y)),test_size=.25,stratify=y,random_state=0)
    mod=mk(m).fit(X[tr],y[tr]); r_rand=ev(y[te],mod.predict(X[te])); del mod; gc.collect()
    folds=[]; oof=np.zeros(len(y),int); t0=time.time()
    for k in range(4):
        trm=G!=k; mod=mk(m).fit(X[trm],y[trm]); p=mod.predict(X[~trm]); oof[~trm]=p; folds.append(ev(y[~trm],p)); del mod; gc.collect()
    mf=[f['macroF1'] for f in folds]
    out[f'{m}|{fs}']={'random_split':r_rand,'file_disjoint_folds':folds,'file_disjoint_macroF1_mean':float(np.mean(mf)),'file_disjoint_macroF1_sd':float(np.std(mf,ddof=1)),
                      'file_disjoint_pooled':ev(y,oof),'f1_per_class_mean':np.mean([f['f1_per_class'] for f in folds],0).tolist()}
    if fs=='noIAT_noTTL': np.save(os.path.join(OUT,f'oof_{m}.npy'),oof)
    print(m,fs,'random',round(r_rand['macroF1'],3),'file-disjoint',round(np.mean(mf),3),'±',round(np.std(mf,ddof=1),3),[round(x,2) for x in out[f'{m}|{fs}']['f1_per_class_mean']],round(time.time()-t0),flush=True)
    json.dump(out,open(os.path.join(OUT,'e1_external.json'),'w'),indent=1,default=float)
