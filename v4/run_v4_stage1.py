#!/usr/bin/env python3
"""Independent V4 Stage-1 runner: raw slices -> durable occupied-voxel score artifacts."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd
from pandas.errors import EmptyDataError
from v4_realtime_core import setup_torch,load_v4_model,load_calibration,build_sparse_item_from_dataframe,predict_v4_sparse_rows,extract_center_metadata
from v4_stage_contracts import CONTRACT_VERSION,STAGE1_MANIFEST_COLUMNS,atomic_json,load_stage1_artifact,save_stage1_artifact,safe_id,stage1_paths,upsert_manifest_row
from run_v4_realtime_session import discover

def pa():
    p=argparse.ArgumentParser(); p.add_argument('--input_dir',required=True); p.add_argument('--session_filter',required=True); p.add_argument('--output_dir',required=True); p.add_argument('--model_path',required=True); p.add_argument('--calibration_json',required=True)
    p.add_argument('--grid_size',type=int,nargs=3,default=[400,400,200]); p.add_argument('--core_size',type=int,default=48); p.add_argument('--batch_size',type=int,default=12); p.add_argument('--amp',choices=['bf16','fp16','none'],default='bf16'); p.add_argument('--compile_model',type=int,default=0)
    p.add_argument('--evaluate_all_cores',type=int,default=1); p.add_argument('--gpu_coord_channels',type=int,default=0); p.add_argument('--fixed_batch_shape',type=int,default=0); p.add_argument('--resume',type=int,default=1); p.add_argument('--max_slices',type=int,default=0); return p.parse_args()

def completed_manifest_keys(path):
    if not path.is_file() or not path.stat().st_size:return set()
    try:d=pd.read_csv(path)
    except EmptyDataError:return set()
    if not {'group_id','slice_seq','status'}<=set(d.columns):return set()
    d=d[d.status.astype(str)=='completed']; return {(str(r.group_id),int(r.slice_seq)) for r in d.itertuples(index=False)}

def main():
    a=pa(); inp=Path(a.input_dir).resolve(); out=Path(a.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True); rows=discover(inp,a.session_filter); rows=rows[:a.max_slices] if a.max_slices>0 else rows
    geo,sess=a.session_filter.split('/',1); manifest_path=out/'stage1_manifest.csv'; done=completed_manifest_keys(manifest_path)
    setup_torch(); model,cfg,compiled=load_v4_model(a.model_path,'cuda',bool(a.compile_model),'default'); cal=load_calibration(a.calibration_json)
    for i,(seq,src,rel) in enumerate(rows,1):
        sid=safe_id(rel.replace('/','__')); npz,meta_path=stage1_paths(out,rel)
        if a.resume and (a.session_filter,seq) in done and npz.is_file() and meta_path.is_file(): print(f'[stage1] {i}/{len(rows)} reuse seq={seq} artifact={npz}',flush=True); continue
        # If data was atomically committed immediately before a crash, repair the missing manifest without repeating GPU inference.
        if a.resume and npz.is_file() and meta_path.is_file():
            _,_,meta=load_stage1_artifact(npz,meta_path); center=dict(meta.get('center_metadata',{})); row={'contract_version':CONTRACT_VERSION,'id':str(meta.get('id',sid)),'source':str(meta.get('source',src)),'relative_path':rel,'geography':geo,'session':sess,'slice_seq':seq,'group_id':a.session_filter,'center_x':center.get('center_x',np.nan),'center_y':center.get('center_y',np.nan),'center_z':center.get('center_z',np.nan),'stage1_npz':str(npz),'stage1_meta_json':str(meta_path),'rows':int(meta.get('rows',0)),'occupied_rows':int(meta.get('occupied_rows',0)),'status':'completed'}; upsert_manifest_row(manifest_path,row,STAGE1_MANIFEST_COLUMNS); done.add((a.session_filter,seq)); print(f'[stage1] {i}/{len(rows)} repaired_manifest seq={seq}',flush=True); continue
        t0=time.perf_counter(); df=pd.read_csv(src); read_ms=(time.perf_counter()-t0)*1000.0; t0=time.perf_counter(); item=build_sparse_item_from_dataframe(df,a.grid_size); prep_ms=(time.perf_counter()-t0)*1000.0
        t0=time.perf_counter(); pred=predict_v4_sparse_rows(item,model,cfg,cal,a.grid_size,a.core_size,a.batch_size,a.amp,evaluate_all_cores=bool(a.evaluate_all_cores),gpu_coord_channels=bool(a.gpu_coord_channels),fixed_batch_shape=bool(a.fixed_batch_shape)); infer_ms=(time.perf_counter()-t0)*1000.0
        center=extract_center_metadata(df); timing={'csv_read_ms':read_ms,'sparse_item_prep_ms':prep_ms,'stage1_wall_ms':infer_ms,**pred['timing']}; meta={'id':sid,'source':str(src),'relative_path':rel,'geography':geo,'session':sess,'slice_seq':seq,'group_id':a.session_filter,'rows':len(df),'occupied_rows':len(item['coords']),'center_metadata':center,'timing':timing,'model_path':str(a.model_path),'calibration_json':str(a.calibration_json),'compiled':compiled,'evaluate_all_cores':bool(a.evaluate_all_cores),'gpu_coord_channels':bool(a.gpu_coord_channels),'fixed_batch_shape':bool(a.fixed_batch_shape),'amp':a.amp}
        t0=time.perf_counter(); save_stage1_artifact(npz,meta_path,item,pred,meta); write_ms=(time.perf_counter()-t0)*1000.0; row={'contract_version':CONTRACT_VERSION,'id':sid,'source':str(src),'relative_path':rel,'geography':geo,'session':sess,'slice_seq':seq,'group_id':a.session_filter,**center,'stage1_npz':str(npz),'stage1_meta_json':str(meta_path),'rows':len(df),'occupied_rows':len(item['coords']),'status':'completed'}; upsert_manifest_row(manifest_path,row,STAGE1_MANIFEST_COLUMNS); done.add((a.session_filter,seq)); print(f"[stage1] {i}/{len(rows)} seq={seq} occupied={len(item['coords'])} infer={infer_ms:.1f}ms artifact_write={write_ms:.1f}ms",flush=True)
    atomic_json({'completed':True,'contract_version':CONTRACT_VERSION,'stage':1,'group_id':a.session_filter,'slices':len(rows),'manifest':str(manifest_path)},out/'STAGE1_COMPLETED.json')
if __name__=='__main__':main()
