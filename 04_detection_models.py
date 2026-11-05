# Main revised protocol: dedup, IAT excluded, pseudo-session blocked 5-fold CV, nested randomised tuning
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from common import *
import sys, pickle, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, matthews_corrcoef, confusion_matrix
import lightgbm as lgb, xgboost as xgb
dd=pd.read_parquet(os.path.join(DATA,'dedup.parquet')); X=dd[FEATS_NOIAT].to_numpy(np.float32); y=dd.sub_label.values; B=dd.block.values
iat=dd.IAT.values
which=sys.argv[1].split(',')
SPACES={
 'DT':(DecisionTreeClassifier(random_state=0),{'max_depth':[None,10,15,20,30],'min_samples_leaf':[1,2,5,10,20,50],'criterion':['gini','entropy'],'class_weight':[None,'balanced']}),
 'RF':(RandomForestClassifier(random_state=0,n_jobs=1),{'n_estimators':[100,200],'max_depth':[None,15,25],'min_samples_leaf':[1,2,5,10],'max_features':['sqrt',0.5],'class_weight':[None,'balanced_subsample']}),
 'LGBM':(lgb.LGBMClassifier(random_state=0,verbose=-1,n_jobs=1),{'n_estimators':[200,400,800],'learning_rate':[0.03,0.05,0.1],'num_leaves':[31,63,127,255],'min_child_samples':[10,30,100],
        'subsample':[0.7,1.0],'subsample_freq':[1],'colsample_bytree':[0.6,0.8,1.0],'reg_lambda':[0,1,5],'class_weight':[None,'balanced']}),
 'XGB':(xgb.XGBClassifier(random_state=0,tree_method='hist',n_jobs=1,verbosity=0),{'n_estimators':[200,400,800],'learning_rate':[0.05,0.1,0.2],'max_depth':[6,8,10,12],
        'min_child_weight':[1,5],'subsample':[0.7,1.0],'colsample_bytree':[0.6,0.8,1.0],'reg_lambda':[1,5]}),
}
def fixed(n):
    if n=='Dummy': return DummyClassifier(strategy='stratified',random_state=0)
    if n=='LR': return make_pipeline(QuantileTransformer(output_distribution='normal',subsample=200000,random_state=0),LogisticRegression(max_iter=500,class_weight='balanced',C=1.0))
    if n=='MLP': return make_pipeline(QuantileTransformer(output_distribution='normal',subsample=200000,random_state=0),MLPClassifier(hidden_layer_sizes=(128,64),early_stopping=True,max_iter=40,batch_size=1024,random_state=0))
def ev(yt,p): return {'acc':accuracy_score(yt,p),'bal_acc':balanced_accuracy_score(yt,p),'macroF1':f1_score(yt,p,average='macro'),'mcc':matthews_corrcoef(yt,p),
    'f1_per_class':f1_score(yt,p,average=None).tolist(),'cm':confusion_matrix(yt,p).tolist()}
rng=np.random.default_rng(0)
CONT=[i for i,f in enumerate(FEATS_NOIAT) if f!='Protocol Type']
for n in which:
    out={'folds':[],'params':[]}; oof_p=np.zeros(len(y),int); oof_pr=np.zeros((len(y),6),np.float32)
    for k in range(5):
        tr=np.where(B!=k)[0]; te=np.where(B==k)[0]
        if n in SPACES:
            est,space=SPACES[n]
            # inner subsample, inner blocked 3-fold within training fold
            sub=pd.Series(tr).groupby(y[tr]).apply(lambda s: s.sample(min(len(s),8000),random_state=k)).values
            r=pd.Series(iat[sub]).groupby(y[sub]).rank(method='first',pct=True).values
            ps=PredefinedSplit(np.minimum((r*3).astype(int),2))
            yy=y[sub]
            rs=RandomizedSearchCV(est,space,n_iter=15,scoring='f1_macro',cv=ps,random_state=k,n_jobs=2,refit=False).fit(X[sub],yy)
            bp=rs.best_params_; out['params'].append({**bp,'inner_cv_macroF1':rs.best_score_})
            m=type(est)(**{**est.get_params(),**bp}); 
            if 'n_jobs' in m.get_params(): m.set_params(n_jobs=2)
        else: m=fixed(n)
        t=time.time(); m.fit(X[tr],y[tr]); tt=time.time()-t
        t=time.time(); pr=m.predict_proba(X[te]); it=time.time()-t; p=pr.argmax(1)
        oof_p[te]=p; oof_pr[te]=pr
        f=ev(y[te],p); f.update(train_s=tt,infer_s_per_1k=1000*it/len(te))
        # robustness: multiplicative log-normal noise on continuous features
        if n in ('RF','LGBM','XGB','DT'):
            f['noise']={}
            for s in [0.05,0.1,0.2,0.5]:
                Xn=X[te].copy(); Xn[:,CONT]*=np.exp(rng.normal(0,s,(len(te),len(CONT)))).astype(np.float32)
                f['noise'][s]=f1_score(y[te],m.predict(Xn),average='macro')
            fp=f'{RES}/_tmp.pkl'; pickle.dump(m,open(fp,'wb')); f['model_MB']=os.path.getsize(fp)/1e6
            if k==0 and n in ('LGBM','RF','XGB'): os.replace(fp,f'{RES}/model_{n}_fold0.pkl')
            elif os.path.exists(fp): os.remove(fp)
        out['folds'].append(f); print(n,k,round(f['macroF1'],4),round(tt,1),flush=True)
        save(f's3_{n}',out); del m; import gc; gc.collect()
    out['oof']=ev(y,oof_p); np.save(f'{RES}/oof_{n}.npy',oof_pr); save(f's3_{n}',out)
