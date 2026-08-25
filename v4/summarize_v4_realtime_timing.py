#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

DEFAULT_COLUMNS = [
 'csv_read_ms','sparse_item_prep_ms','stage1_total_ms','stage1_wall_ms',
 'patch_build_ms','host_batch_pack_ms','host_pin_ms','h2d_ms','sparse_h2d_cuda_ms',
 'gpu_sparse_scatter_ms','gpu_patch_extract_ms','gpu_feature_assembly_ms','gpu_model_ms','gpu_gather_ms','d2h_gather_ms',
 'stage1_artifact_load_ms','stage1_artifact_write_ms','stage1_manifest_write_ms',
 'stage2_component_ms','stage2_refiner_parametric_ms','stage2_total_ms','stage2_artifact_write_ms','stage2_manifest_write_ms',
 'stage3_incremental_ms','stage3_algorithm_ms','stage3_fragment_join_ms','stage3_span_completion_pre_ms','stage3_hidden_pole_ms',
 'stage3_span_completion_post_ms','stage3_chain_build_attachment_ms','stage3_output_write_ms','stage3_wrapper_overhead_ms',
 'arrival_to_publish_ms','end_to_end_update_ms'
]


def one(a):
    a = pd.to_numeric(a, errors='coerce').dropna().to_numpy(float)
    if len(a) == 0: return None
    return {'count':int(len(a)),'mean_ms':float(a.mean()),'p50_ms':float(np.quantile(a,.5)),
            'p95_ms':float(np.quantile(a,.95)),'min_ms':float(a.min()),'max_ms':float(a.max())}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--timing_csv',required=True); ap.add_argument('--output_json'); ap.add_argument('--output_txt'); a=ap.parse_args()
    p=Path(a.timing_csv)
    if not p.is_file(): raise FileNotFoundError(f'Timing CSV not found: {p}')
    try: d=pd.read_csv(p)
    except EmptyDataError as exc: raise RuntimeError(f'Timing CSV has no columns: {p}') from exc
    if d.empty: raise RuntimeError(f'Timing CSV contains zero rows: {p}')
    report={'timing_csv':str(p),'rows':len(d),'columns':{},'stage3_by_window_observed_slices':{},'runtime_modes':{}}
    for c in DEFAULT_COLUMNS:
        if c in d.columns:
            z=one(d[c])
            if z: report['columns'][c]=z
    if {'stage3_window_observed_slices','stage3_incremental_ms'}<=set(d.columns):
        tmp=d[['stage3_window_observed_slices','stage3_incremental_ms']].apply(pd.to_numeric,errors='coerce').dropna()
        for k,g in tmp.groupby('stage3_window_observed_slices'): report['stage3_by_window_observed_slices'][str(int(k))]=one(g['stage3_incremental_ms'])
    for c in ('gpu_volume_mode','fixed_batch_shape'):
        if c in d.columns:
            report['runtime_modes'][c] = {str(k):int(v) for k,v in d[c].value_counts(dropna=False).to_dict().items()}
    lines=[f'V4 production timing summary: {p}',f'slices={len(d)}','']
    for c,z in report['columns'].items(): lines.append(f"{c:36s} mean={z['mean_ms']:10.3f} p50={z['p50_ms']:10.3f} p95={z['p95_ms']:10.3f} min={z['min_ms']:10.3f} max={z['max_ms']:10.3f}")
    if report['stage3_by_window_observed_slices']:
        lines += ['','Stage3 wall time by observed centers in [S-9,S] (maximum 10 centers):']
        for k,z in sorted(report['stage3_by_window_observed_slices'].items(),key=lambda kv:int(kv[0])): lines.append(f"  centers={int(k):2d} mean={z['mean_ms']:10.3f} p50={z['p50_ms']:10.3f} p95={z['p95_ms']:10.3f} n={z['count']}")
    txt='\n'.join(lines)+'\n'; print(txt,end='')
    if a.output_json: q=Path(a.output_json); q.parent.mkdir(parents=True,exist_ok=True); q.write_text(json.dumps(report,indent=2))
    if a.output_txt: q=Path(a.output_txt); q.parent.mkdir(parents=True,exist_ok=True); q.write_text(txt)


if __name__=='__main__': main()
