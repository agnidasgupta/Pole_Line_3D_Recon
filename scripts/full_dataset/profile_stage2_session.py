#!/usr/bin/env python3
"""Run accepted V4 Stage 2 logic from saved Stage 1 artifacts with timing CSV."""
from __future__ import annotations
import argparse,time
from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError
from v4_realtime_pipeline import V4Stage2Processor
from v4_stage2_runtime import POLE_OUTPUT_COLUMNS,LINE_OUTPUT_COLUMNS,VERTEX_OUTPUT_COLUMNS
from v4_stage_contracts import CONTRACT_VERSION,STAGE1_MANIFEST_COLUMNS,STAGE2_MANIFEST_COLUMNS,atomic_csv,atomic_json,load_stage1_artifact,stage1_paths,stage2_paths,upsert_manifest_row

def pa():
 p=argparse.ArgumentParser(); p.add_argument('--stage1_dir',required=True); p.add_argument('--output_dir',required=True); p.add_argument('--session_filter',required=True); p.add_argument('--stage2_bundle',required=True); p.add_argument('--timing_csv',required=True); p.add_argument('--resume',type=int,default=1); p.add_argument('--grid_size',type=int,nargs=3,default=[400,400,200]); p.add_argument('--voxel_size_ft',type=float,default=.5); p.add_argument('--pole_candidate_threshold',type=float,default=.15); p.add_argument('--line_candidate_threshold',type=float,default=.08); p.add_argument('--line_weak_threshold',type=float,default=.04); p.add_argument('--line_competition_ratio',type=float,default=.55); p.add_argument('--pole_min_voxels',type=int,default=4); p.add_argument('--line_min_voxels',type=int,default=3); p.add_argument('--edge_width_vox',type=int,default=10); return p.parse_args()

def main():
 a=pa(); s1root=Path(a.stage1_dir).resolve(); out=Path(a.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True); tp=Path(a.timing_csv).resolve(); tp.parent.mkdir(parents=True,exist_ok=True); mp=s1root/'stage1_manifest.csv'
 if not mp.is_file():raise FileNotFoundError(mp)
 try:man=pd.read_csv(mp)
 except EmptyDataError as e:raise RuntimeError(f'Stage1 manifest has no rows: {mp}') from e
 miss=sorted(set(STAGE1_MANIFEST_COLUMNS)-set(man.columns));
 if miss:raise RuntimeError(f'Stage1 manifest missing columns {miss}: {mp}')
 man=man[(man.group_id.astype(str)==a.session_filter)&(man.status.astype(str)=='completed')].copy().sort_values('slice_seq')
 if man.empty:raise RuntimeError(f'No completed Stage1 slices for {a.session_filter}')
 proc=V4Stage2Processor(a.stage2_bundle,a.grid_size,a.voxel_size_ft,a.pole_candidate_threshold,a.line_candidate_threshold,a.line_weak_threshold,a.line_competition_ratio,a.pole_min_voxels,a.line_min_voxels,a.edge_width_vox); out_manifest=out/'inference_manifest.csv'
 try:existing=pd.read_csv(out_manifest) if out_manifest.is_file() and out_manifest.stat().st_size else pd.DataFrame()
 except EmptyDataError:existing=pd.DataFrame()
 done={(str(r.group_id),int(r.slice_seq)) for r in existing.itertuples(index=False) if hasattr(r,'status') and str(r.status)=='completed'} if not existing.empty and {'group_id','slice_seq','status'}<=set(existing.columns) else set(); timings=[]
 for i,r in enumerate(man.itertuples(index=False),1):
  seq=int(r.slice_seq); polecsv,linecsv,vertcsv=stage2_paths(out,str(r.relative_path)); t_all=time.perf_counter()
  if a.resume and (a.session_filter,seq) in done and polecsv.is_file() and linecsv.is_file() and vertcsv.is_file():
   timings.append({'group_id':a.session_filter,'slice_seq':seq,'stage2_load_ms':0.0,'stage2_total_ms':0.0,'stage2_artifact_write_ms':0.0,'stage2_manifest_write_ms':0.0,'stage2_wall_ms':0.0,'resume_from':'stage2'}); print(f'[stage2] {i}/{len(man)} reuse seq={seq}',flush=True); continue
  s1_npz=Path(str(r.stage1_npz)); s1_meta=Path(str(r.stage1_meta_json))
  if not s1_npz.is_file() or not s1_meta.is_file():
   fallback_npz,fallback_meta=stage1_paths(s1root,str(r.relative_path));
   if fallback_npz.is_file() and fallback_meta.is_file():s1_npz,s1_meta=fallback_npz,fallback_meta
  t0=time.perf_counter(); item,pred,meta=load_stage1_artifact(s1_npz,s1_meta); load_ms=(time.perf_counter()-t0)*1000.0
  t0=time.perf_counter(); s2=proc.process(item,pred,str(r.id),seq); stage2_ms=(time.perf_counter()-t0)*1000.0
  t0=time.perf_counter(); atomic_csv(s2['poles'],polecsv,POLE_OUTPUT_COLUMNS); atomic_csv(s2['lines'],linecsv,LINE_OUTPUT_COLUMNS); atomic_csv(s2['vertices'],vertcsv,VERTEX_OUTPUT_COLUMNS); write_ms=(time.perf_counter()-t0)*1000.0
  row={'contract_version':CONTRACT_VERSION,'id':str(r.id),'source':str(r.source),'relative_path':str(r.relative_path),'geography':str(r.geography),'session':str(r.session),'slice_seq':seq,'group_id':str(r.group_id),'center_x':float(r.center_x),'center_y':float(r.center_y),'center_z':float(r.center_z),'stage1_npz':str(s1_npz),'output_csv':'','pole_csv':str(polecsv),'line_csv':str(linecsv),'line_vertices_csv':str(vertcsv),'rows':int(r.rows),'accepted_poles':len(s2['poles']),'accepted_line_segments':len(s2['lines']),'status':'completed'}
  t0=time.perf_counter(); d=upsert_manifest_row(out_manifest,row,STAGE2_MANIFEST_COLUMNS); atomic_csv(d,out/'stage2_manifest.csv',STAGE2_MANIFEST_COLUMNS); manifest_ms=(time.perf_counter()-t0)*1000.0; done.add((a.session_filter,seq)); wall=(time.perf_counter()-t_all)*1000.0
  timings.append({'group_id':a.session_filter,'slice_seq':seq,'stage2_load_ms':load_ms,'stage2_total_ms':stage2_ms,'stage2_artifact_write_ms':write_ms,'stage2_manifest_write_ms':manifest_ms,'stage2_wall_ms':wall,'accepted_poles':len(s2['poles']),'accepted_line_segments':len(s2['lines']),'resume_from':'none'}); print(f'[stage2] {i}/{len(man)} seq={seq} poles={len(s2["poles"])} lines={len(s2["lines"])} load={load_ms:.1f}ms stage2={stage2_ms:.1f}ms wall={wall:.1f}ms',flush=True)
 pd.DataFrame(timings).to_csv(tp,index=False); atomic_json({'completed':True,'contract_version':CONTRACT_VERSION,'stage':2,'group_id':a.session_filter,'slices':len(man),'manifest':str(out_manifest),'timing_csv':str(tp)},out/'STAGE2_COMPLETED.json')
if __name__=='__main__':main()
