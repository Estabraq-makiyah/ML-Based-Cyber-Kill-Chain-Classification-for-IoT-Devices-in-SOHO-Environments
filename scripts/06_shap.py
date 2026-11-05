import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from common import *
import pickle, shap
dd=pd.read_parquet(os.path.join(DATA,'dedup.parquet')); X=dd[FEATS_NOIAT].astype(np.float32); y=dd.sub_label.values; B=dd.block.values
m=pickle.load(open(f'{RES}/model_LGBM_fold0.pkl','rb'))
tr=np.where(B!=0)[0]; te=np.where(B==0)[0]
bg=pd.Series(tr).groupby(y[tr]).apply(lambda s: s.sample(200,random_state=0)).values      # 1,200 background rows
ex=pd.Series(te).groupby(y[te]).apply(lambda s: s.sample(min(len(s),500),random_state=1)).values  # up to 3,000 explained rows
e=shap.TreeExplainer(m,feature_perturbation='tree_path_dependent')
sv=e.shap_values(X.iloc[ex],check_additivity=False)
sv=np.array(sv); 
if sv.shape[0]!=6: sv=np.moveaxis(sv,-1,0)   # -> (class, n, feat)
mean_abs=np.abs(sv).mean(axis=(0,1)); per_class=np.abs(sv).mean(axis=1)
gain=m.booster_.feature_importance('gain'); split=m.booster_.feature_importance('split')
out={'mode':'tree_path_dependent','n_background':0,'n_explained':len(ex),'seeds':{'background':0,'explained':1},'features':FEATS_NOIAT,
     'mean_abs_shap':mean_abs.tolist(),'per_class_mean_abs_shap':per_class.tolist(),'gain':gain.tolist(),'split':split.tolist()}
save('s5_shap',out); np.save(f'{RES}/shap_vals.npy',sv); np.save(f'{RES}/shap_X.npy',X.iloc[ex].to_numpy()); np.save(f'{RES}/shap_y.npy',y[ex])
print(sorted(zip(mean_abs,FEATS_NOIAT))[::-1][:8])
