#!/usr/bin/env python3
"""Mine Stage-2 components entirely in slice-local coordinates."""
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
import pandas as pd, torch
from pandas.errors import EmptyDataError
from voxel_common import load_npz_dense,read_json
from v6_common import safe_id
from v6_components import atomic_json,extract_all_components
from v6_predict import load_v6_model,predict_dense_scores,setup_torch
from v62_local import add_edge_features,assign_noise_tolerant_targets

def pa():
 p=argparse.ArgumentParser(); p.add_argument('--dataset_dir',required=True); p.add_argument('--model_path',required=True); p.add_argument('--output_dir',required=True)
 p.add_argument('--splits',default='train,val'); p.add_argument('--pole_candidate_threshold',type=float,default=.15); p.add_argument('--line_candidate_threshold',type=float,default=.08); p.add_argument('--line_weak_threshold',type=float,default=.04); p.add_argument('--line_competition_ratio',type=float,default=.55); p.add_argument('--line_min_voxels',type=int,default=3)
 p.add_argument('--voxel_size_ft',type=float,default=.5); p.add_argument('--edge_width_vox',type=int,default=10); p.add_argument('--core_size',type=int,default=48)
 p.add_argument('--batch_size',type=int,default=5); p.add_argument('--build_workers',type=int,default=6); p.add_argument('--amp',default='bf16'); p.add_argument('--compile_model',type=int,default=1); p.add_argument('--resume',type=int,default=1)
 return p.parse_args()

def main():
 a=pa(); setup_torch(); out=Path(a.output_dir); done=out/'done'; done.mkdir(parents=True,exist_ok=True)
 summary=read_json(os.path.join(a.dataset_dir,'manifests','summary.json')); grid=tuple(summary['grid_size_xyz'])
 model,cfg,compiled=load_v6_model(a.model_path,compile_model=bool(a.compile_model)); print('GPU:',torch.cuda.get_device_name(0),'compile:',compiled)
 recs=[]
 for split in [x.strip() for x in a.splits.split(',') if x.strip()]:
  for r in read_json(os.path.join(a.dataset_dir,'manifests',split+'.json')): q=dict(r); q['split']=split; recs.append(q)
 frames=[]; started=time.time()
 for i,r in enumerate(recs,1):
  sid=safe_id(r); op=done/(sid+'.csv.gz')
  if a.resume and op.exists():
   try: d=pd.read_csv(op)
   except EmptyDataError: d=pd.DataFrame(columns=['split','file_id','group_id','source_relpath','global_component_key'])
  else:
   item=load_npz_dense(r['npz_path'],grid,use_dist=bool(int(cfg.get('use_dist',1))))
   scores=predict_dense_scores(item,model,cfg,grid,a.core_size,a.batch_size,a.build_workers,a.amp,bool(int(cfg.get('channels_last',1))),float(cfg.get('score_sem_weight',.55)),float(cfg.get('score_binary_weight',.35)),float(cfg.get('score_object_weight',.10)))
   comps=extract_all_components(item['occ'],scores,a.pole_candidate_threshold,a.line_candidate_threshold,a.voxel_size_ft,item['labels'],line_min_voxels=a.line_min_voxels,attachment_radius_ft=9999.,line_weak_threshold=a.line_weak_threshold,line_competition_ratio=a.line_competition_ratio)
   fs=[]
   for name in ('poles','lines'):
    x=comps[name].copy()
    if x.empty: continue
    x=add_edge_features(x,comps['pole_points' if name=='poles' else 'line_points'],grid,a.edge_width_vox)
    x=assign_noise_tolerant_targets(x,'pole' if name=='poles' else 'line')
    x.insert(0,'split',r['split']); x.insert(1,'file_id',sid); x.insert(2,'group_id',r.get('group_id','')); x.insert(3,'source_relpath',r.get('source_relpath',''))
    x['global_component_key']=[f"{sid}|{c}" for c in x.component_id]
    fs.append(x)
   d=pd.concat(fs,ignore_index=True) if fs else pd.DataFrame(columns=['split','file_id','group_id','source_relpath','global_component_key']); op.parent.mkdir(parents=True,exist_ok=True); d.to_csv(op,index=False,compression='gzip')
  if not d.empty: frames.append(d)
  atomic_json({'completed_files':i,'total_files':len(recs),'percent':100*i/max(len(recs),1),'last_file':sid,'elapsed_seconds':time.time()-started},out/'progress.json')
  print(f'[stage2-mine] {i}/{len(recs)} {sid} rows={len(d)}')
 allc=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
 if allc.empty: raise RuntimeError('No local components mined')
 allc.to_csv(out/'all_local_components.csv.gz',index=False,compression='gzip')
 counts=allc.groupby(['split','class_name','target','target_source']).size().reset_index(name='count'); counts.to_csv(out/'target_counts.csv',index=False)
 atomic_json({'completed':True,'files':len(recs),'components':len(allc),'target_counts':counts.to_dict('records'),'elapsed_seconds':time.time()-started},out/'COMPLETED.json')
 print(counts.to_string(index=False))
if __name__=='__main__': main()
