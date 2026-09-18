#!/usr/bin/env python3
"""V4 Stage-1 + Stage-2 runtime equivalence and latency gate.

Reference: original full-core CPU patch construction.  Optimized variants must preserve
Stage-1 labels/scores AND Stage-2 accepted object topology, parametric geometry and
refiner probabilities before they are eligible for production selection.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd
from v4_realtime_core import setup_torch,load_v4_model,load_calibration,build_sparse_item_from_dataframe,predict_v4_sparse_rows,label_from_scores
from v4_realtime_pipeline import V4Stage2Processor

VARIANTS={
 'full_cpu':dict(evaluate_all_cores=True,gpu_coord_channels=False,fixed_batch_shape=False),
 'active_cpu':dict(evaluate_all_cores=False,gpu_coord_channels=False,fixed_batch_shape=True),
 'full_gpu':dict(evaluate_all_cores=True,gpu_coord_channels=True,fixed_batch_shape=False),
 'active_gpu':dict(evaluate_all_cores=False,gpu_coord_channels=True,fixed_batch_shape=True),
}

def run_one(item,model,cfg,cal,batch,amp,variant):
    t=time.perf_counter(); out=predict_v4_sparse_rows(item,model,cfg,cal,batch_size=batch,amp=amp,**VARIANTS[variant]); wall=time.perf_counter()-t
    lab=label_from_scores(out['pole'],out['line'],cal['pole_threshold'],cal['line_threshold']); return out,lab,wall

def _keys(df,cols):
    if df is None or df.empty:return []
    return [tuple(x) for x in df.sort_values(cols,kind='stable')[cols].astype(str).to_numpy()]

def _numeric_max(ref,cur,key_cols,ignore=()):
    if ref is None or ref.empty:
        return 0.0 if cur is None or cur.empty else float('inf')
    if cur is None or cur.empty:return float('inf')
    r=ref.sort_values(key_cols,kind='stable').reset_index(drop=True); c=cur.sort_values(key_cols,kind='stable').reset_index(drop=True)
    if _keys(r,key_cols)!=_keys(c,key_cols):return float('inf')
    nums=[x for x in r.columns if x in c.columns and x not in set(key_cols)|set(ignore) and pd.api.types.is_numeric_dtype(r[x]) and pd.api.types.is_numeric_dtype(c[x])]
    mx=0.0
    for col in nums:
        a=pd.to_numeric(r[col],errors='coerce').to_numpy(float); b=pd.to_numeric(c[col],errors='coerce').to_numpy(float)
        finite=np.isfinite(a)&np.isfinite(b)
        if np.any(np.isfinite(a)!=np.isfinite(b)):return float('inf')
        if finite.any():mx=max(mx,float(np.max(np.abs(a[finite]-b[finite]))))
    return mx

def stage2_compare(ref,cur):
    rp,cp=ref['poles'],cur['poles']; rl,cl=ref['lines'],cur['lines']; rv,cv=ref['vertices'],cur['vertices']
    pole_keys=['component_id']; line_keys=['component_id']; vert_keys=['component_id','vertex_index']
    topology=(_keys(rp,pole_keys)==_keys(cp,pole_keys) and _keys(rl,line_keys)==_keys(cl,line_keys) and _keys(rv,vert_keys)==_keys(cv,vert_keys))
    geom=max(_numeric_max(rp,cp,pole_keys,ignore=('refiner_probability','slice_seq')),_numeric_max(rl,cl,line_keys,ignore=('refiner_probability','slice_seq')),_numeric_max(rv,cv,vert_keys,ignore=('slice_seq',)))
    prob=max(_numeric_max(rp,cp,pole_keys,ignore=tuple(x for x in rp.columns if x!='refiner_probability')) if not rp.empty else 0.0,_numeric_max(rl,cl,line_keys,ignore=tuple(x for x in rl.columns if x!='refiner_probability')) if not rl.empty else 0.0)
    # _numeric_max with all other numeric columns ignored isolates refiner_probability.
    return {'topology_match':bool(topology),'geometry_max_abs':float(geom),'refiner_probability_max_abs':float(prob),'poles_ref':len(rp),'poles_cur':len(cp),'lines_ref':len(rl),'lines_cur':len(cl),'vertices_ref':len(rv),'vertices_cur':len(cv)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input_dir',required=True); ap.add_argument('--max_files',type=int,default=32); ap.add_argument('--model_path',required=True); ap.add_argument('--calibration_json',required=True); ap.add_argument('--stage2_bundle',required=True); ap.add_argument('--batch_size',type=int,default=12); ap.add_argument('--amp',choices=['bf16','fp16','none'],default='bf16'); ap.add_argument('--score_tol',type=float,default=1e-4); ap.add_argument('--stage2_geometry_tol',type=float,default=1e-6); ap.add_argument('--stage2_probability_tol',type=float,default=1e-5); ap.add_argument('--output',required=True); a=ap.parse_args()
    paths=sorted([*Path(a.input_dir).rglob('*.csv'),*Path(a.input_dir).rglob('*.csv.gz')]);
    if not paths:raise FileNotFoundError(a.input_dir)
    n=min(max(1,a.max_files),len(paths)); idx=np.linspace(0,len(paths)-1,n,dtype=int); paths=[paths[i] for i in np.unique(idx)]
    setup_torch(); model,cfg,_=load_v4_model(a.model_path,'cuda',False); cal=load_calibration(a.calibration_json); s2proc=V4Stage2Processor(a.stage2_bundle); rows=[]
    for i,path in enumerate(paths,1):
        df=pd.read_csv(path); item=build_sparse_item_from_dataframe(df); ref,ref_lab,ref_s=run_one(item,model,cfg,cal,a.batch_size,a.amp,'full_cpu'); ref_s2=s2proc.process(item,ref,path.stem,i)
        rec={'csv':str(path),'occupied_voxels':len(item['coords']),'reference_seconds':ref_s,'full_cores':ref['timing']['active_cores']}
        for name in ('active_cpu','full_gpu','active_gpu'):
            out,lab,wall=run_one(item,model,cfg,cal,a.batch_size,a.amp,name); dp=np.abs(out['pole']-ref['pole']); dl=np.abs(out['line']-ref['line']); cur_s2=s2proc.process(item,out,path.stem,i); s2cmp=stage2_compare(ref_s2,cur_s2)
            rec[name]={'seconds':wall,'speedup_vs_full_cpu':ref_s/max(wall,1e-9),'cores':out['timing']['active_cores'],'pole_max_abs':float(dp.max(initial=0)),'line_max_abs':float(dl.max(initial=0)),'pole_mean_abs':float(dp.mean()) if len(dp) else 0.0,'line_mean_abs':float(dl.mean()) if len(dl) else 0.0,'stage1_label_mismatches':int(np.sum(lab!=ref_lab)),'stage2':s2cmp,'timing':out['timing']}
        rows.append(rec); ag=rec['active_gpu']; fg=rec['full_gpu']; print(f"[v4-gate] {i}/{len(paths)} full_gpu speedup={fg['speedup_vs_full_cpu']:.3f} s1_mismatch={fg['stage1_label_mismatches']} s2_topology={fg['stage2']['topology_match']} | active_gpu speedup={ag['speedup_vs_full_cpu']:.3f} s1_mismatch={ag['stage1_label_mismatches']} s2_topology={ag['stage2']['topology_match']}",flush=True)
    summary={}
    for name in ('active_cpu','full_gpu','active_gpu'):
        vals=[r[name] for r in rows]; summary[name]={'pole_max_abs':max(v['pole_max_abs'] for v in vals),'line_max_abs':max(v['line_max_abs'] for v in vals),'stage1_label_mismatches':sum(v['stage1_label_mismatches'] for v in vals),'stage2_topology_mismatches':sum(not v['stage2']['topology_match'] for v in vals),'stage2_geometry_max_abs':max(v['stage2']['geometry_max_abs'] for v in vals),'stage2_refiner_probability_max_abs':max(v['stage2']['refiner_probability_max_abs'] for v in vals),'speedup_mean':float(np.mean([v['speedup_vs_full_cpu'] for v in vals])),'speedup_p50':float(np.median([v['speedup_vs_full_cpu'] for v in vals]))}
        s=summary[name]; s['pass_recommended']=bool(s['pole_max_abs']<=a.score_tol and s['line_max_abs']<=a.score_tol and s['stage1_label_mismatches']==0 and s['stage2_topology_mismatches']==0 and s['stage2_geometry_max_abs']<=a.stage2_geometry_tol and s['stage2_refiner_probability_max_abs']<=a.stage2_probability_tol)
    passing=[n for n in ('active_gpu','active_cpu','full_gpu') if summary[n]['pass_recommended']]; recommended=max(passing,key=lambda n:summary[n]['speedup_mean']) if passing else 'full_cpu'; report={'files':len(rows),'amp':a.amp,'batch_size':a.batch_size,'score_tol':a.score_tol,'stage2_geometry_tol':a.stage2_geometry_tol,'stage2_probability_tol':a.stage2_probability_tol,'reference':'full_cpu','summary':summary,'recommended_runtime':recommended,'per_file':rows}; op=Path(a.output);op.parent.mkdir(parents=True,exist_ok=True);op.write_text(json.dumps(report,indent=2));print(json.dumps({'recommended_runtime':recommended,'summary':summary},indent=2))
if __name__=='__main__':main()
