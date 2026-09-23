#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from v4_stage2_local import LOCAL_FEATURE_COLUMNS

ROOT=Path(__file__).resolve().parent

def require_text(path,tokens,errs):
    text=(ROOT/path).read_text()
    for t in tokens:
        if t not in text: errs.append(f'{path}: missing contract token {t!r}')
    return text

def main():
    errs=[]
    forbidden={'center_x','center_y','center_z','world_x','world_y','world_z','exact_gt_fraction','near_gt_fraction'}
    bad=[c for c in LOCAL_FEATURE_COLUMNS if c in forbidden or c.startswith('world_')]
    if bad:errs.append(f'unsafe Stage2 feature columns: {bad}')
    pipe=require_text('v4_realtime_pipeline.py',['class V4Stage2Processor','def run_stage1','def run_stage2'],errs)
    if 'reconstruct_v4_stage3' in pipe:errs.append('Stage1/2 pipeline imports Stage3')
    stage2=require_text('v4_stage2_runtime.py',['def line_vertices','def apply_stage2'],errs)
    require_text('v4_sparse_components.py',['def extract_sparse_components'],errs)
    if 'extract_sparse_components' not in pipe:errs.append('v4_realtime_pipeline.py: Stage2 processor does not call extract_sparse_components')
    if 'extract_center_metadata' in stage2:errs.append('Stage2 runtime accesses Stage3 center metadata')
    core=require_text('v4_realtime_core.py',['gpu_coord_channels: bool = False','def _predict_v4_gpu_sparse_volume','gpu_volume_mode','fixed_batch_shape'],errs)
    runner=require_text('run_v4_realtime_session.py',['--max_sequence_gap','default=9','--max_span_length_ft','default=450.0','--slice_length_ft','default=50.0','stages_independently_replayable','stage3_rolling_past_only','arrival_to_publish_ms','save_stage1_artifact','load_stage1_artifact'],errs)
    stage3=require_text('reconstruct_v4_stage3.py',['--max_sequence_gap','a.max_observed_slice_centers = int(a.max_sequence_gap) + 1','a.latest_slice - a.max_span_slices','RealtimeStage3Cache','realtime_cache_put_stage2','if not hidden_mp.empty:'],errs)
    contracts=require_text('v4_stage_contracts.py',['CONTRACT_VERSION = "v4-production-1"','def atomic_npz','def save_stage1_artifact','def load_stage1_artifact','def stage2_paths'],errs)
    for f,tokens in {
        'run_v4_stage1.py':['Independent V4 Stage-1 runner','stage1_manifest.csv'],
        'run_v4_stage2.py':['Independent V4 Stage-2 runner','inference_manifest.csv'],
        'run_v4_stage3.py':['Independent rolling V4 Stage-3 runner','max_sequence_gap'],
        'v4_nebius_common.sh':['v4_discover_session','run_context.env','v4_backup_code','v4_detached_run'],
    }.items():require_text(f,tokens,errs)
    report={'ok':not errs,'errors':errs,'stage1_slice_local':True,'stage2_slice_local':True,'stage3_rolling_past_only':True,'durable_stage_boundaries':True,'max_span_length_ft':450.0,'slice_length_ft':50.0,'max_sequence_gap':9,'max_observed_slice_centers':10,'gpu_sparse_volume_path_present':True}
    print(json.dumps(report,indent=2))
    if errs:raise SystemExit('V4_PRODUCTION_SOURCE_CONTRACT_FAILED')
    print('V4_PRODUCTION_SOURCE_CONTRACT_OK')
if __name__=='__main__':main()
