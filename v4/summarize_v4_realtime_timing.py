#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_COLUMNS = [
    'csv_read_ms','sparse_item_prep_ms','patch_build_ms','h2d_ms','gpu_feature_assembly_ms',
    'gpu_model_ms','d2h_gather_ms','stage1_wall_ms','stage2_component_ms',
    'stage2_refiner_parametric_ms','stage12_wall_ms','stage3_incremental_ms',
    'stage3_algorithm_ms','stage3_fragment_join_ms','stage3_span_completion_pre_ms',
    'stage3_hidden_pole_ms','stage3_span_completion_post_ms','stage3_chain_build_attachment_ms',
    'stage3_output_write_ms','stage3_wrapper_overhead_ms','end_to_end_update_ms'
]

def one(a):
    a=pd.to_numeric(a,errors='coerce').dropna().to_numpy(float)
    if len(a)==0:return None
    return {'count':int(len(a)),'mean_ms':float(a.mean()),'p50_ms':float(np.quantile(a,.5)),'p95_ms':float(np.quantile(a,.95)),'min_ms':float(a.min()),'max_ms':float(a.max())}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--timing_csv',required=True)
    ap.add_argument('--output_json',default=None)
    ap.add_argument('--output_txt',default=None)
    a=ap.parse_args()
    p=Path(a.timing_csv); d=pd.read_csv(p)
    report={'timing_csv':str(p),'rows':len(d),'columns':{},'stage3_by_window_observed_slices':{}}
    for c in DEFAULT_COLUMNS:
        if c in d.columns:
            z=one(d[c])
            if z: report['columns'][c]=z
    if 'stage3_window_observed_slices' in d.columns and 'stage3_incremental_ms' in d.columns:
        tmp=d[['stage3_window_observed_slices','stage3_incremental_ms']].copy()
        tmp['stage3_window_observed_slices']=pd.to_numeric(tmp['stage3_window_observed_slices'],errors='coerce')
        tmp['stage3_incremental_ms']=pd.to_numeric(tmp['stage3_incremental_ms'],errors='coerce')
        tmp=tmp.dropna()
        for k,g in tmp.groupby('stage3_window_observed_slices'):
            report['stage3_by_window_observed_slices'][str(int(k))]=one(g['stage3_incremental_ms'])
    lines=[f'V4 realtime timing summary: {p}',f'slices={len(d)}','']
    for c,z in report['columns'].items():
        lines.append(f"{c:34s} mean={z['mean_ms']:10.3f}  p50={z['p50_ms']:10.3f}  p95={z['p95_ms']:10.3f}  min={z['min_ms']:10.3f}  max={z['max_ms']:10.3f}")
    if report['stage3_by_window_observed_slices']:
        lines += ['', 'Stage3 incremental wall time by observed slices in rolling window:']
        for k,z in sorted(report['stage3_by_window_observed_slices'].items(),key=lambda kv:int(kv[0])):
            lines.append(f"  window_slices={int(k):2d}  mean={z['mean_ms']:10.3f}  p50={z['p50_ms']:10.3f}  p95={z['p95_ms']:10.3f}  n={z['count']}")
    txt='\n'.join(lines)+'\n'
    print(txt,end='')
    if a.output_json:
        q=Path(a.output_json);q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(report,indent=2))
    if a.output_txt:
        q=Path(a.output_txt);q.parent.mkdir(parents=True,exist_ok=True);q.write_text(txt)
if __name__=='__main__':main()
