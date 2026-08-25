#!/usr/bin/env python3
"""Latency sweep for an already gated V4 runtime; batch promotion also preserves Stage2."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd
from v4_realtime_core import setup_torch,load_v4_model,load_calibration,build_sparse_item_from_dataframe,predict_v4_sparse_rows,label_from_scores
from v4_realtime_pipeline import V4Stage2Processor
from compare_v4_runtime_variants import stage2_compare

MODES={'full_cpu':dict(evaluate_all_cores=True,gpu_coord_channels=False,fixed_batch_shape=False),'active_cpu':dict(evaluate_all_cores=False,gpu_coord_channels=False,fixed_batch_shape=True),'full_gpu':dict(evaluate_all_cores=True,gpu_coord_channels=True,fixed_batch_shape=False),'active_gpu':dict(evaluate_all_cores=False,gpu_coord_channels=True,fixed_batch_shape=True)}
def stat(a):
 a=np.asarray(a,float); return {'mean_ms':float(a.mean()),'p50_ms':float(np.quantile(a,.5)),'p95_ms':float(np.quantile(a,.95)),'min_ms':float(a.min()),'max_ms':float(a.max())}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--input_dir',required=True);ap.add_argument('--model_path',required=True);ap.add_argument('--calibration_json',required=True);ap.add_argument('--stage2_bundle',required=True);ap.add_argument('--runtime_mode',choices=MODES,required=True);ap.add_argument('--batch_sizes',default='8,12,16,20,24,32');ap.add_argument('--reference_batch',type=int,default=12);ap.add_argument('--max_files',type=int,default=8);ap.add_argument('--amp',choices=['bf16','fp16','none'],default='bf16');ap.add_argument('--score_tol',type=float,default=1e-4);ap.add_argument('--stage2_geometry_tol',type=float,default=1e-6);ap.add_argument('--stage2_probability_tol',type=float,default=1e-5);ap.add_argument('--output_dir',required=True);a=ap.parse_args()
 bs=[int(x) for x in a.batch_sizes.split(',') if x.strip()];
 if a.reference_batch not in bs:bs.append(a.reference_batch)
 bs=sorted(set(bs)); paths=sorted([*Path(a.input_dir).rglob('*.csv'),*Path(a.input_dir).rglob('*.csv.gz')]);
 if not paths:raise FileNotFoundError(a.input_dir)
 n=min(a.max_files,len(paths));idx=np.linspace(0,len(paths)-1,n,dtype=int);paths=[paths[i] for i in np.unique(idx)];setup_torch();model,cfg,_=load_v4_model(a.model_path,'cuda',False);cal=load_calibration(a.calibration_json);mode=MODES[a.runtime_mode];s2proc=V4Stage2Processor(a.stage2_bundle);items=[]
 for p in paths:items.append((p,build_sparse_item_from_dataframe(pd.read_csv(p))))
 refs={}
 for path,item in items:
  out=predict_v4_sparse_rows(item,model,cfg,cal,batch_size=a.reference_batch,amp=a.amp,**mode);lab=label_from_scores(out['pole'],out['line'],cal['pole_threshold'],cal['line_threshold']);s2=s2proc.process(item,out,path.stem,0);refs[str(path)]=(out['pole'].copy(),out['line'].copy(),lab.copy(),s2)
 rows=[];summary=[]
 for batch in bs:
  predict_v4_sparse_rows(items[0][1],model,cfg,cal,batch_size=batch,amp=a.amp,**mode);total=[];parts={k:[] for k in ['host_pin_ms','sparse_h2d_cuda_ms','gpu_sparse_scatter_ms','gpu_patch_extract_ms','gpu_feature_assembly_ms','gpu_model_ms','gpu_gather_ms','d2h_gather_ms']};maxp=maxl=maxgeom=maxprob=0.0;mism=topomism=0
  for path,item in items:
   t=time.perf_counter();out=predict_v4_sparse_rows(item,model,cfg,cal,batch_size=batch,amp=a.amp,**mode);wall=(time.perf_counter()-t)*1000;lab=label_from_scores(out['pole'],out['line'],cal['pole_threshold'],cal['line_threshold']);cur_s2=s2proc.process(item,out,path.stem,0);rp,rl,rla,rs2=refs[str(path)];maxp=max(maxp,float(np.max(np.abs(out['pole']-rp),initial=0)));maxl=max(maxl,float(np.max(np.abs(out['line']-rl),initial=0)));mism+=int(np.sum(lab!=rla));cmp=stage2_compare(rs2,cur_s2);topomism+=int(not cmp['topology_match']);maxgeom=max(maxgeom,cmp['geometry_max_abs']);maxprob=max(maxprob,cmp['refiner_probability_max_abs']);rec={'batch_size':batch,'csv':str(path),'occupied_voxels':len(item['coords']),'stage1_wall_ms':wall,**out['timing']};rows.append(rec);total.append(wall)
   for k in parts:
    if k in out['timing']:parts[k].append(float(out['timing'][k]))
  ok=bool(maxp<=a.score_tol and maxl<=a.score_tol and mism==0 and topomism==0 and maxgeom<=a.stage2_geometry_tol and maxprob<=a.stage2_probability_tol);sm={'batch_size':batch,'reference_batch':a.reference_batch,'runtime_mode':a.runtime_mode,'files':len(items),'stage1_wall':stat(total),'pole_max_abs_vs_reference_batch':maxp,'line_max_abs_vs_reference_batch':maxl,'label_mismatches_vs_reference_batch':mism,'stage2_topology_mismatches':topomism,'stage2_geometry_max_abs':maxgeom,'stage2_refiner_probability_max_abs':maxprob,'pass_equivalence':ok}
  for k,v in parts.items():
   if v:sm[k]=stat(v)
  summary.append(sm);print(json.dumps(sm,indent=2),flush=True)
 outdir=Path(a.output_dir);outdir.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(outdir/'batch_size_per_file.csv',index=False);(outdir/'batch_size_summary.json').write_text(json.dumps(summary,indent=2));pd.DataFrame([{'batch_size':x['batch_size'],'runtime_mode':x['runtime_mode'],'mean_ms':x['stage1_wall']['mean_ms'],'p50_ms':x['stage1_wall']['p50_ms'],'p95_ms':x['stage1_wall']['p95_ms'],'label_mismatches':x['label_mismatches_vs_reference_batch'],'stage2_topology_mismatches':x['stage2_topology_mismatches'],'pass_equivalence':x['pass_equivalence']} for x in summary]).to_csv(outdir/'batch_size_summary.csv',index=False)
if __name__=='__main__':main()
