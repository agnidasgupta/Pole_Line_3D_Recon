#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from v4_realtime_core import setup_torch,load_v4_model,load_calibration,build_sparse_item_from_dataframe,predict_v4_sparse_rows,label_from_scores

MODES={
 'full_cpu':dict(evaluate_all_cores=True,gpu_coord_channels=False),
 'active_cpu':dict(evaluate_all_cores=False,gpu_coord_channels=False),
 'full_gpu':dict(evaluate_all_cores=True,gpu_coord_channels=True),
 'active_gpu':dict(evaluate_all_cores=False,gpu_coord_channels=True),
}

def stat(a):
 a=np.asarray(a,float)
 return {'mean_ms':float(a.mean()),'p50_ms':float(np.quantile(a,.5)),'p95_ms':float(np.quantile(a,.95)),'min_ms':float(a.min()),'max_ms':float(a.max())}

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--input_dir',required=True); ap.add_argument('--model_path',required=True); ap.add_argument('--calibration_json',required=True)
 ap.add_argument('--runtime_mode',choices=MODES,default='full_cpu'); ap.add_argument('--batch_sizes',default='8,12,16,20'); ap.add_argument('--max_files',type=int,default=8)
 ap.add_argument('--amp',choices=['bf16','fp16','none'],default='bf16'); ap.add_argument('--score_tol',type=float,default=1e-4); ap.add_argument('--output_dir',required=True)
 a=ap.parse_args(); bs=[int(x) for x in a.batch_sizes.split(',') if x.strip()]
 if not bs: raise ValueError('no batch sizes')
 paths=sorted([*Path(a.input_dir).rglob('*.csv'),*Path(a.input_dir).rglob('*.csv.gz')])
 if not paths: raise FileNotFoundError(a.input_dir)
 n=min(a.max_files,len(paths)); idx=np.linspace(0,len(paths)-1,n,dtype=int); paths=[paths[i] for i in np.unique(idx)]
 setup_torch(); model,cfg,_=load_v4_model(a.model_path,'cuda',False); cal=load_calibration(a.calibration_json); mode=MODES[a.runtime_mode]
 items=[]
 for p in paths:
  df=pd.read_csv(p); items.append((p,build_sparse_item_from_dataframe(df)))
 refs={}; rows=[]; summary=[]
 for bi,batch in enumerate(bs):
  # warm current batch configuration once; excluded from measured rows
  predict_v4_sparse_rows(items[0][1],model,cfg,cal,batch_size=batch,amp=a.amp,**mode)
  total=[]; parts={k:[] for k in ['patch_build_ms','h2d_ms','gpu_feature_assembly_ms','gpu_model_ms','d2h_gather_ms']}
  maxp=maxl=0.0; mism=0
  for path,item in items:
   t=time.perf_counter(); out=predict_v4_sparse_rows(item,model,cfg,cal,batch_size=batch,amp=a.amp,**mode); wall=(time.perf_counter()-t)*1000
   lab=label_from_scores(out['pole'],out['line'],cal['pole_threshold'],cal['line_threshold']); key=str(path)
   if bi==0: refs[key]=(out['pole'].copy(),out['line'].copy(),lab.copy())
   else:
    rp,rl,rla=refs[key]; maxp=max(maxp,float(np.max(np.abs(out['pole']-rp),initial=0))); maxl=max(maxl,float(np.max(np.abs(out['line']-rl),initial=0))); mism+=int(np.sum(lab!=rla))
   rec={'batch_size':batch,'csv':key,'occupied_voxels':len(item['coords']),'stage1_wall_ms':wall,**out['timing']}; rows.append(rec); total.append(wall)
   for k in parts: parts[k].append(float(out['timing'][k]))
  sm={'batch_size':batch,'runtime_mode':a.runtime_mode,'files':len(items),'stage1_wall':stat(total),'pole_max_abs_vs_reference_batch':maxp,'line_max_abs_vs_reference_batch':maxl,'label_mismatches_vs_reference_batch':mism,'pass_equivalence':bool(maxp<=a.score_tol and maxl<=a.score_tol and mism==0)}
  for k,v in parts.items(): sm[k]=stat(v)
  summary.append(sm); print(json.dumps(sm,indent=2),flush=True)
 outdir=Path(a.output_dir); outdir.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(rows).to_csv(outdir/'batch_size_per_file.csv',index=False)
 (outdir/'batch_size_summary.json').write_text(json.dumps(summary,indent=2))
 flat=[]
 for x in summary:
  flat.append({'batch_size':x['batch_size'],'runtime_mode':x['runtime_mode'],'mean_ms':x['stage1_wall']['mean_ms'],'p50_ms':x['stage1_wall']['p50_ms'],'p95_ms':x['stage1_wall']['p95_ms'],'label_mismatches':x['label_mismatches_vs_reference_batch'],'pole_max_abs':x['pole_max_abs_vs_reference_batch'],'line_max_abs':x['line_max_abs_vs_reference_batch'],'pass_equivalence':x['pass_equivalence']})
 pd.DataFrame(flat).to_csv(outdir/'batch_size_summary.csv',index=False)
if __name__=='__main__':main()
