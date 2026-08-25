#!/usr/bin/env python3
"""Offline mining for V4 realtime Stage 2.

Stage 1 and Stage 2 operate on one slice at a time. GT is consulted only after
V4 inference to construct offline refiner targets; it is never a model feature.
"""
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
import numpy as np,pandas as pd,torch

from v4_realtime_core import setup_torch,load_v4_model,load_calibration,build_sparse_item_from_npz,predict_v4_sparse_rows
from v4_sparse_components import extract_sparse_components
from v4_stage2_local import LOCAL_FEATURE_COLUMNS


def atomic_json(x,p):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(x,indent=2)); os.replace(q,p)


def args():
    p=argparse.ArgumentParser(); p.add_argument('--dataset_dir',required=True); p.add_argument('--output_dir',required=True)
    p.add_argument('--model_path',required=True); p.add_argument('--calibration_json',required=True)
    p.add_argument('--grid_size',type=int,nargs=3,default=[400,400,200]); p.add_argument('--voxel_size_ft',type=float,default=.5)
    p.add_argument('--core_size',type=int,default=48); p.add_argument('--batch_size',type=int,default=12); p.add_argument('--amp',choices=['fp16','bf16','none'],default='bf16')
    p.add_argument('--pole_candidate_threshold',type=float,default=.15); p.add_argument('--line_candidate_threshold',type=float,default=.08)
    p.add_argument('--line_weak_threshold',type=float,default=.04); p.add_argument('--line_competition_ratio',type=float,default=.55)
    p.add_argument('--pole_min_voxels',type=int,default=4); p.add_argument('--line_min_voxels',type=int,default=3); p.add_argument('--edge_width_vox',type=int,default=10)
    p.add_argument('--compile_model',type=int,default=0); p.add_argument('--evaluate_all_cores',type=int,default=1); p.add_argument('--gpu_coord_channels',type=int,default=0); p.add_argument('--resume',type=int,default=1); p.add_argument('--max_files',type=int,default=0)
    return p.parse_args()


def main():
    a=args(); setup_torch(); out=Path(a.output_dir); per=out/'per_slice'; per.mkdir(parents=True,exist_ok=True)
    ds=Path(a.dataset_dir); manifests=ds/'manifests'; records=[]
    for split in ('train','val'):
        q=json.load(open(manifests/f'{split}.json'))
        for r in q: records.append((split,r))
    if a.max_files>0: records=records[:a.max_files]
    model,cfg,compiled=load_v4_model(a.model_path,'cuda',bool(a.compile_model),'default'); cal=load_calibration(a.calibration_json)
    print(f'[v4-stage2-mine] files={len(records)} compiled={compiled} amp={a.amp}')
    timings=[]; started=time.time(); all_paths=[]
    for i,(split,r) in enumerate(records,1):
        sid=str(r['id']); op=per/split/f'{sid}.csv'; op.parent.mkdir(parents=True,exist_ok=True); all_paths.append(op)
        if a.resume and op.is_file() and op.stat().st_size>0:
            print(f'[v4-stage2-mine] {i}/{len(records)} reuse {split}/{sid}'); continue
        t0=time.perf_counter(); item=build_sparse_item_from_npz(r['npz_path'],a.grid_size); load_ms=(time.perf_counter()-t0)*1000
        gt={1:item['coords'][item['raw_labels']==1],2:item['coords'][item['raw_labels']==2]}
        t0=time.perf_counter(); pred=predict_v4_sparse_rows(item,model,cfg,cal,a.grid_size,a.core_size,a.batch_size,a.amp,evaluate_all_cores=bool(a.evaluate_all_cores),gpu_coord_channels=bool(a.gpu_coord_channels)); pred_ms=(time.perf_counter()-t0)*1000
        t0=time.perf_counter(); comps=extract_sparse_components(item,pred,a.grid_size,a.voxel_size_ft,a.pole_candidate_threshold,a.line_candidate_threshold,
                    a.line_weak_threshold,a.line_competition_ratio,a.pole_min_voxels,a.line_min_voxels,gt,a.edge_width_vox); comp_ms=(time.perf_counter()-t0)*1000
        frames=[]
        for name in ('poles','lines'):
            d=comps[name].copy()
            if not d.empty:
                d['split']=split; d['file_id']=sid; d['group_id']=r.get('group_id',''); d['npz_path']=r['npz_path']; frames.append(d)
        frame=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=['component_id','class_name','target','target_weight','target_source','split','file_id','group_id'])
        tmp=op.with_suffix('.csv.tmp'); frame.to_csv(tmp,index=False); os.replace(tmp,op)
        tt=pred['timing']; timings.append({'file_id':sid,'split':split,'occupied_voxels':len(item['coords']),'components':len(frame),'load_ms':load_ms,
            'patch_build_ms':tt['patch_build_ms'],'h2d_ms':tt['h2d_ms'],'gpu_feature_assembly_ms':tt.get('gpu_feature_assembly_ms',0.0),'gpu_model_ms':tt['gpu_model_ms'],'d2h_gather_ms':tt['d2h_gather_ms'],
            'component_ms':comp_ms,'total_ms':load_ms+pred_ms+comp_ms,'active_cores':tt['active_cores'],'total_possible_cores':tt['total_possible_cores']})
        if i%25==0 or i==len(records): print(f'[v4-stage2-mine] {i}/{len(records)} elapsed={time.time()-started:.1f}s')
    dfs=[]
    for p in all_paths:
        if p.is_file() and p.stat().st_size>0:
            try: dfs.append(pd.read_csv(p))
            except Exception as e: raise RuntimeError(f'Cannot read cached component file {p}: {e}')
    if not dfs: raise RuntimeError('No Stage-2 components were mined')
    allc=pd.concat(dfs,ignore_index=True); allc.to_csv(out/'components.csv',index=False)
    if timings: pd.DataFrame(timings).to_csv(out/'stage2_mining_runtime.csv',index=False)
    counts=allc.groupby(['split','class_name','target'],dropna=False).size().reset_index(name='rows'); counts.to_csv(out/'target_counts.csv',index=False)
    report={'completed':True,'files':len(records),'component_rows':len(allc),'compile_model':bool(a.compile_model),'evaluate_all_cores':bool(a.evaluate_all_cores),'gpu_coord_channels':bool(a.gpu_coord_channels),'amp':a.amp,
            'stage1_stage2_slice_local':True,'gt_features_in_model':False,'feature_columns':LOCAL_FEATURE_COLUMNS,
            'note':'GT overlap is used only to assign offline Stage-2 targets. Runtime Stage 1/2 uses one slice and no center/world/session features.'}
    atomic_json(report,out/'COMPLETED.json'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
