#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd


def stats(path,cols):
    p=Path(path)
    if not p.is_file(): return {'missing':str(p)}
    d=pd.read_csv(p); out={'rows':len(d)}
    for c in cols:
        if c not in d: continue
        a=pd.to_numeric(d[c],errors='coerce').dropna().to_numpy(float)
        if len(a): out[c]={'mean':float(a.mean()),'p50':float(np.quantile(a,.5)),'p95':float(np.quantile(a,.95)),'max':float(a.max())}
    return out

def readj(p):
    try:return json.load(open(p))
    except Exception:return None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run_root',required=True); ap.add_argument('--replay_dir',default=None); ap.add_argument('--output',required=True); a=ap.parse_args()
    r=Path(a.run_root); report={'stage2_training_completed':readj(r/'STAGE2_TRAINING_COMPLETED.json'),'stage2_refiner_metrics':readj(r/'stage2_refiner/local_refiner_metrics.json'),
        'stage2_mining_timing':stats(r/'stage2_mining/stage2_mining_runtime.csv',['load_ms','patch_build_ms','h2d_ms','gpu_feature_assembly_ms','gpu_model_ms','d2h_gather_ms','component_ms','total_ms'])}
    if a.replay_dir:
        q=Path(a.replay_dir); report['replay_completed']=readj(q/'COMPLETED.json'); report['realtime_timing']=stats(q/'realtime_slice_timing.csv',['csv_read_ms','sparse_item_prep_ms','patch_build_ms','h2d_ms','gpu_feature_assembly_ms','gpu_model_ms','d2h_gather_ms','stage1_wall_ms','stage2_component_ms','stage2_refiner_parametric_ms','stage12_wall_ms','stage3_incremental_ms','stage3_algorithm_ms','stage3_fragment_join_ms','stage3_span_completion_pre_ms','stage3_hidden_pole_ms','stage3_span_completion_post_ms','stage3_chain_build_attachment_ms','stage3_output_write_ms','stage3_wrapper_overhead_ms','end_to_end_update_ms'])
        latest=q/'stage3_incremental'
        completed=list(latest.rglob('COMPLETED.json')) if latest.exists() else []; report['incremental_stage3_snapshots']=len(completed)
    Path(a.output).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__':main()
