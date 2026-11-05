# Phase 2 reworked: equivalence proof, merged-stage, hierarchical cascade, held-out-subtype stage generalisation,
# leave-one-attack-family-out detection, cost-sensitive stage decisions.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from common import *
import warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, recall_score, confusion_matrix
import lightgbm as lgb
from collections import Counter
dd=pd.read_parquet(os.path.join(DATA,'dedup.parquet')); X=dd[FEATS_NOIAT].to_numpy(np.float32); y=dd.sub_label.values; B=dd.block.values
STAGE={0:0,1:1,2:1,3:2,4:3,5:4}   # 0 benign, 1 delivery(DDoS+DoS), 2/3/4 = single-category stages (classes 3,4,5)
ys=np.vectorize(STAGE.get)(y)
P=json.load(open(f'{RES}/s3_LGBM.json'))['params']
def params(k):
    p={kk:v for kk,v in P[k].items() if kk!='inner_cv_macroF1'}; return {**p,'random_state':0,'verbose':-1,'n_jobs':2}
def cap(idx,lab,n=100000,seed=0):
    return pd.Series(idx).groupby(lab[idx]).apply(lambda s: s.sample(min(len(s),n),random_state=seed)).values
out={}
# (a) equivalence: phase-1 OOF collapsed to stages vs flat stage model
oof1=np.load(f'{RES}/oof_LGBM.npy')
p1_stage=np.vectorize(STAGE.get)(oof1.argmax(1))
flat=np.zeros(len(y),int); hier=np.zeros(len(y),int); det=np.zeros(len(y),int)
for k in range(5):
    tr=np.where(B!=k)[0]; te=np.where(B==k)[0]
    m=lgb.LGBMClassifier(**params(k)).fit(X[tr],ys[tr]); flat[te]=m.predict(X[te])
    # (b) hierarchical: binary detector then stage classifier trained on attacks only
    mb=lgb.LGBMClassifier(**params(k)).fit(X[tr],(y[tr]>0).astype(int)); d=mb.predict(X[te]); det[te]=d
    tra=tr[y[tr]>0]; ms=lgb.LGBMClassifier(**params(k)).fit(X[tra],ys[tra])
    h=np.zeros(len(te),int); h[d==1]=ms.predict(X[te][d==1]); hier[te]=h
    print('fold',k,flush=True)
def ev(a,b): return {'macroF1':f1_score(a,b,average='macro'),'bal_acc':balanced_accuracy_score(a,b),'acc':accuracy_score(a,b),'f1_per_stage':f1_score(a,b,average=None).tolist(),'cm':confusion_matrix(a,b).tolist()}
out['phase1_collapsed_to_stage']=ev(ys,p1_stage)
out['flat_stage_model']=ev(ys,flat)
out['agreement_flat_vs_phase1_collapsed']=float((flat==p1_stage).mean())
out['hierarchical']=ev(ys,hier)
out['binary_detector']={'TPR':recall_score(y>0,det),'FPR':float(((det==1)&(y==0)).sum()/(y==0).sum())}
# (c) held-out Delivery subtype: does an unseen DoS (or DDoS) flow land in the Delivery stage?
out['heldout_subtype']={}
for hold in [1,2]:
    res=[]
    for k in range(5):
        tr=np.where((B!=k)&(y!=hold))[0]; tr=cap(tr,y,seed=k); te=np.where((B==k)&(y==hold))[0]
        m=lgb.LGBMClassifier(**params(k)).fit(X[tr],ys[tr]); p=m.predict(X[te])
        res.append({s:float((p==s).mean()) for s in range(5)})
    out['heldout_subtype'][hold]=pd.DataFrame(res).mean().to_dict(); print('hold',hold,out['heldout_subtype'][hold],flush=True)
# (d) leave-one-attack-family-out binary detection (open-set / unseen-family robustness)
out['lofo_detection']={}
for c in [1,2,3,4,5]:
    res=[]
    for k in range(5):
        tr=np.where((B!=k)&(y!=c))[0]; tr=cap(tr,y,seed=k); te=np.where(B==k)[0]
        m=lgb.LGBMClassifier(**params(k)).fit(X[tr],(y[tr]>0).astype(int)); p=m.predict(X[te])
        res.append({'recall_unseen_family':float(p[y[te]==c].mean()),'FPR_benign':float(p[y[te]==0].mean()),
                    'recall_seen_attacks':float(p[(y[te]>0)&(y[te]!=c)].mean())})
    out['lofo_detection'][c]=pd.DataFrame(res).mean().to_dict(); print('lofo',c,out['lofo_detection'][c],flush=True)
# (e) cost-sensitive stage decision from phase-1 OOF probabilities
Ps=np.zeros((len(y),5)); 
for c,s in STAGE.items(): Ps[:,s]+=oof1[:,c]
C=np.full((5,5),2.0); np.fill_diagonal(C,0); C[1:,0]=10.0; C[0,1:]=1.0   # miss=10, false alarm=1, wrong stage=2
def cost(pred): return float(C[ys,pred].mean())
argmax=Ps.argmax(1); bayes=(Ps@C).argmin(1)   # expected cost of predicting j = sum_i P(i) C[i,j]
out['cost']={'matrix':C.tolist(),'argmax':{'mean_cost':cost(argmax),'missed_attacks':int(((argmax==0)&(ys>0)).sum()),'false_alarms':int(((argmax>0)&(ys==0)).sum()),**ev(ys,argmax)},
             'bayes_min_cost':{'mean_cost':cost(bayes),'missed_attacks':int(((bayes==0)&(ys>0)).sum()),'false_alarms':int(((bayes>0)&(ys==0)).sum()),**ev(ys,bayes)}}
save('s4_phase2',out); print(json.dumps({k:v for k,v in out.items() if k in('agreement_flat_vs_phase1_collapsed','cost')},default=float)[:1500])
