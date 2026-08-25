#!/usr/bin/env python3
"""Four-way V4 Stage-1 runtime equivalence gate.

Reference is full 48^3 core tiling + original CPU-generated coordinate channels.
Variants isolate active-core scheduling from GPU coordinate generation.
Stage 1 remains strictly one-slice-local in every case.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from v4_realtime_core import setup_torch, load_v4_model, load_calibration, build_sparse_item_from_dataframe, predict_v4_sparse_rows, label_from_scores

VARIANTS = {
    'full_cpu':   dict(evaluate_all_cores=True,  gpu_coord_channels=False),
    'active_cpu': dict(evaluate_all_cores=False, gpu_coord_channels=False),
    'full_gpu':   dict(evaluate_all_cores=True,  gpu_coord_channels=True),
    'active_gpu': dict(evaluate_all_cores=False, gpu_coord_channels=True),
}

def run_one(item, model, cfg, cal, batch, amp, variant):
    kw=VARIANTS[variant]
    t=time.perf_counter()
    out=predict_v4_sparse_rows(item,model,cfg,cal,batch_size=batch,amp=amp,**kw)
    wall=time.perf_counter()-t
    lab=label_from_scores(out['pole'],out['line'],cal['pole_threshold'],cal['line_threshold'])
    return out, lab, wall

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input_dir',required=True)
    ap.add_argument('--max_files',type=int,default=32)
    ap.add_argument('--model_path',required=True)
    ap.add_argument('--calibration_json',required=True)
    ap.add_argument('--batch_size',type=int,default=12)
    ap.add_argument('--amp',choices=['bf16','fp16','none'],default='bf16')
    ap.add_argument('--score_tol',type=float,default=1e-4)
    ap.add_argument('--output',required=True)
    a=ap.parse_args()

    paths=sorted([*Path(a.input_dir).rglob('*.csv'),*Path(a.input_dir).rglob('*.csv.gz')])
    if not paths: raise FileNotFoundError(a.input_dir)
    n=min(max(1,a.max_files),len(paths))
    idx=np.linspace(0,len(paths)-1,n,dtype=int)
    paths=[paths[i] for i in np.unique(idx)]

    setup_torch(); model,cfg,_=load_v4_model(a.model_path,'cuda',False); cal=load_calibration(a.calibration_json)
    rows=[]
    for i,path in enumerate(paths,1):
        df=pd.read_csv(path); item=build_sparse_item_from_dataframe(df)
        ref, ref_lab, ref_s = run_one(item,model,cfg,cal,a.batch_size,a.amp,'full_cpu')
        rec={'csv':str(path),'occupied_voxels':len(item['coords']),'reference_seconds':ref_s,'full_cores':ref['timing']['active_cores']}
        for name in ('active_cpu','full_gpu','active_gpu'):
            out,lab,wall=run_one(item,model,cfg,cal,a.batch_size,a.amp,name)
            dp=np.abs(out['pole']-ref['pole']); dl=np.abs(out['line']-ref['line'])
            rec[name]={
                'seconds':wall,
                'speedup_vs_full_cpu':ref_s/max(wall,1e-9),
                'cores':out['timing']['active_cores'],
                'pole_max_abs':float(dp.max(initial=0)),
                'line_max_abs':float(dl.max(initial=0)),
                'pole_mean_abs':float(dp.mean()) if len(dp) else 0.0,
                'line_mean_abs':float(dl.mean()) if len(dl) else 0.0,
                'stage1_label_mismatches':int(np.sum(lab!=ref_lab)),
            }
        rows.append(rec)
        ac=rec['active_cpu']; ag=rec['active_gpu']
        print(f"[v4-variant] {i}/{len(paths)} active_cpu speedup={ac['speedup_vs_full_cpu']:.3f} mismatch={ac['stage1_label_mismatches']} line_max={ac['line_max_abs']:.6g} | active_gpu line_max={ag['line_max_abs']:.6g}",flush=True)

    summary={}
    for name in ('active_cpu','full_gpu','active_gpu'):
        vals=[r[name] for r in rows]
        summary[name]={
            'pole_max_abs':max(v['pole_max_abs'] for v in vals),
            'line_max_abs':max(v['line_max_abs'] for v in vals),
            'stage1_label_mismatches':sum(v['stage1_label_mismatches'] for v in vals),
            'speedup_mean':float(np.mean([v['speedup_vs_full_cpu'] for v in vals])),
            'speedup_p50':float(np.median([v['speedup_vs_full_cpu'] for v in vals])),
        }
        s=summary[name]
        s['pass_recommended']=bool(s['pole_max_abs']<=a.score_tol and s['line_max_abs']<=a.score_tol and s['stage1_label_mismatches']==0)

    recommended='active_cpu' if summary['active_cpu']['pass_recommended'] else 'full_cpu'
    report={'files':len(rows),'amp':a.amp,'batch_size':a.batch_size,'score_tol':a.score_tol,'reference':'full_cpu','summary':summary,'recommended_runtime':recommended,'per_file':rows}
    op=Path(a.output); op.parent.mkdir(parents=True,exist_ok=True); op.write_text(json.dumps(report,indent=2))
    print(json.dumps({'recommended_runtime':recommended,'summary':summary},indent=2))

if __name__=='__main__': main()
