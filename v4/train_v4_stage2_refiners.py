#!/usr/bin/env python3
"""Train slice-local Stage-2 component refiners with ambiguous GT disagreements excluded."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import joblib,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix,precision_recall_curve,average_precision_score,roc_auc_score
import os

def atomic_json(obj,path):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(obj,indent=2)); os.replace(tmp,path)
from v4_stage2_local import LOCAL_FEATURE_COLUMNS,pole_local_physical_ok,line_local_physical_ok

def pa():
 p=argparse.ArgumentParser(); p.add_argument('--components_csv',required=True); p.add_argument('--output_dir',required=True); p.add_argument('--trees',type=int,default=600); p.add_argument('--max_depth',type=int,default=20); p.add_argument('--min_samples_leaf',type=int,default=2); p.add_argument('--seed',type=int,default=42); p.add_argument('--pole_target_recall',type=float,default=.95); p.add_argument('--line_target_recall',type=float,default=.98); p.add_argument('--target_precision',type=float,default=.80,help='legacy fallback'); p.add_argument('--pole_target_precision',type=float,default=.80); p.add_argument('--line_target_precision',type=float,default=.60); return p.parse_args()
def mets(y,p):
 cm=confusion_matrix(y,p,labels=[0,1]); tn,fp,fn,tp=cm.ravel(); pr=tp/max(tp+fp,1); rc=tp/max(tp+fn,1); i=tp/max(tp+fp+fn,1); return {'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'precision':pr,'recall':rc,'iou':i,'f1':2*pr*rc/max(pr+rc,1e-9),'cm':cm.tolist()}
def choose(y,prob,physical,target_p,target_r):
 best=None; rows=[]
 for t in np.linspace(.02,.98,97):
  pred=(prob>=t)&physical; m=mets(y,pred.astype(int)); gap=max(0,target_p-m['precision'])+2*max(0,target_r-m['recall']); score=4*m['iou']+3*m['recall']+m['precision']-8*gap; row={'threshold':float(t),'score':score,'gap':gap,**{k:v for k,v in m.items() if k!='cm'}}; rows.append(row); best=row if best is None or score>best['score'] else best
 return best,pd.DataFrame(rows)
def plot_pr(y,prob,path,title):
 p,r,_=precision_recall_curve(y,prob); plt.figure(figsize=(7,6)); plt.plot(r,p); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(title); plt.grid(alpha=.3); plt.xlim(0,1); plt.ylim(0,1); plt.tight_layout(); plt.savefig(path,dpi=180); plt.close()
def train(df,name,a,out):
 t_all=time.perf_counter(); d=df[(df.class_name==name)&(df.target>=0)].copy(); tr=d[d.split=='train']; va=d[d.split=='val']
 if tr.target.nunique()<2 or va.target.nunique()<2: raise RuntimeError(f'{name}: need positive and negative train/val targets; counts train={tr.target.value_counts().to_dict()} val={va.target.value_counts().to_dict()}')
 X=lambda q:q.reindex(columns=LOCAL_FEATURE_COLUMNS,fill_value=0).replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float)
 clf=ExtraTreesClassifier(n_estimators=a.trees,max_depth=a.max_depth,min_samples_leaf=a.min_samples_leaf,class_weight='balanced_subsample',n_jobs=-1,random_state=a.seed,max_features='sqrt')
 w=pd.to_numeric(tr.target_weight,errors='coerce').fillna(1).to_numpy(float); t=time.perf_counter(); clf.fit(X(tr),tr.target.to_numpy(int),sample_weight=w); fit_s=time.perf_counter()-t
 t=time.perf_counter(); prob=clf.predict_proba(X(va))[:,1]; predict_s=time.perf_counter()-t; t=time.perf_counter(); physical=np.array([pole_local_physical_ok(r) if name=='pole' else line_local_physical_ok(r) for _,r in va.iterrows()]); physical_s=time.perf_counter()-t
 target_r=a.pole_target_recall if name=='pole' else a.line_target_recall; target_p=a.pole_target_precision if name=='pole' else a.line_target_precision; best,search=choose(va.target.to_numpy(int),prob,physical,target_p,target_r); pred=((prob>=best['threshold'])&physical).astype(int); m=mets(va.target.to_numpy(int),pred)
 m.update({'threshold':best['threshold'],'average_precision':float(average_precision_score(va.target,prob)),'roc_auc':float(roc_auc_score(va.target,prob)),'train_rows':len(tr),'val_rows':len(va),'ambiguous_excluded':int(((df.class_name==name)&(df.target<0)).sum()),'timing_seconds':{'fit':fit_s,'predict_val':predict_s,'physical_gate_val':physical_s,'total':time.perf_counter()-t_all}})
 joblib.dump(clf,out/f'{name}_refiner.joblib'); search.to_csv(out/f'{name}_threshold_search.csv',index=False); plot_pr(va.target,prob,out/f'{name}_precision_recall.png',f'V4 realtime Stage-2 {name} refiner')
 imp=pd.DataFrame({'feature':LOCAL_FEATURE_COLUMNS,'importance':clf.feature_importances_}).sort_values('importance',ascending=False); imp.to_csv(out/f'{name}_feature_importance.csv',index=False); top=imp.head(20).sort_values('importance'); plt.figure(figsize=(9,7)); plt.barh(top.feature,top.importance); plt.tight_layout(); plt.savefig(out/f'{name}_feature_importance.png',dpi=180); plt.close()
 return clf,m

def main():
 a=pa(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); df=pd.read_csv(a.components_csv); pm,pmet=train(df,'pole',a,out); lm,lmet=train(df,'line',a,out)
 bundle={'version':'v4_realtime_stage2_no_gt_features','feature_columns':LOCAL_FEATURE_COLUMNS,'pole_model':pm,'line_model':lm,'pole_threshold':pmet['threshold'],'line_threshold':lmet['threshold'],'pole_metrics':pmet,'line_metrics':lmet}
 joblib.dump(bundle,out/'local_refiner_bundle.joblib'); report={'completed':True,'pole':pmet,'line':lmet,'note':'V4 Stage 2 is strictly slice-local. World/center coordinates, pole attachments, and GT overlap fractions are not model features. GT overlap is used only to construct training targets. Plausible GT disagreements are excluded from training. Line acceptance uses hysteresis candidates plus a loose impossible-geometry gate so short/occluded fragments can survive to Stage 3.'}; atomic_json(report,out/'local_refiner_metrics.json'); (out/'local_refiner_metrics.txt').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
