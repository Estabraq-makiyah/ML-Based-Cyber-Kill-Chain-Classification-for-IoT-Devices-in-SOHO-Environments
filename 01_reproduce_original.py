import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from common import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import lightgbm as lgb
df=load(); X=df[FEATS]; y=df.sub_label
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
# exact-copy overlap
trkeys=set(map(tuple,Xtr.values.round(10)))
seen=np.array([tuple(r) in trkeys for r in Xte.values.round(10)])
out={'test_rows_with_exact_copy_in_train':float(seen.mean()),'per_class_seen':pd.Series(seen).groupby(yte.values).mean().to_dict()}
models={'RF':RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1),
        'DT':DecisionTreeClassifier(random_state=42),
        'LGBM':lgb.LGBMClassifier(n_estimators=100,learning_rate=0.1,random_state=42,verbose=-1)}
for n,m in models.items():
    t=time.time(); m.fit(Xtr,ytr); tt=time.time()-t
    t=time.time(); p=m.predict(Xte); it=time.time()-t
    out[n]={'acc':accuracy_score(yte,p),'macroF1':f1_score(yte,p,average='macro'),
      'acc_seen':accuracy_score(yte[seen],p[seen]),'acc_unseen':accuracy_score(yte[~seen],p[~seen]),
      'per_class_unseen_recall':pd.Series(p[~seen]==yte.values[~seen]).groupby(yte.values[~seen]).mean().to_dict(),
      'report':classification_report(yte,p,output_dict=True),'cm':confusion_matrix(yte,p).tolist(),'train_s':tt,'infer_s':it}
    print(n,out[n]['acc'],out[n]['acc_seen'],out[n]['acc_unseen'],tt,flush=True)
# Phase 2 equivalence: bijective relabel with same seed
kc={0:'Benign',1:'S3_DDoS',2:'S3_DoS',3:'S1_Recon(orig)',4:'S6_C2(orig)',5:'S4_Exploit(orig)'}
m2=lgb.LGBMClassifier(n_estimators=100,learning_rate=0.1,random_state=42,verbose=-1).fit(Xtr,ytr.map(kc))
p2=m2.predict(Xte); p1=pd.Series(models['LGBM'].predict(Xte)).map(kc).values
out['phase2_identical_to_phase1_mapped']=float((p1==p2).mean())
out['phase2_acc']=accuracy_score(yte.map(kc),p2)
save('s1_reproduce',out); print(out['test_rows_with_exact_copy_in_train'],out['phase2_identical_to_phase1_mapped'])
