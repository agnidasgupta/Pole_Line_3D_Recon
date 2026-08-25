#!/usr/bin/env python3
from __future__ import annotations
import ast, json
from pathlib import Path
from v4_stage2_local import LOCAL_FEATURE_COLUMNS

ROOT=Path(__file__).resolve().parent

def parser_defaults(path,func='args'):
    # lightweight source checks keep the contract visible even before H100 execution
    return Path(path).read_text()

def main():
    errs=[]
    forbidden={'center_x','center_y','center_z','world_x','world_y','world_z','exact_gt_fraction','near_gt_fraction'}
    bad=[c for c in LOCAL_FEATURE_COLUMNS if c in forbidden or c.startswith('world_')]
    if bad: errs.append(f'unsafe Stage2 features: {bad}')
    pipe=(ROOT/'v4_realtime_pipeline.py').read_text()
    if 'reconstruct_v4_stage3' in pipe: errs.append('Stage1/2 pipeline imports Stage3')
    stage2=(ROOT/'v4_stage2_runtime.py').read_text()
    if 'extract_center_metadata' in stage2: errs.append('Stage2 runtime accesses Stage3 center metadata')
    runner=(ROOT/'run_v4_realtime_session.py').read_text()
    required=["default=9", "default=450.", "default=1", "default='inprocess'", "production Stage3 is rolling past-only per arriving slice", "full_session_reconstruction_disabled"]
    for token in required:
        if token not in runner: errs.append(f'missing production default token: {token}')
    stage3=(ROOT/'reconstruct_v4_stage3.py').read_text()
    for token in ['a.latest_slice - a.max_span_slices','--latest_slice requires --session_filter','default=450.0','default=9']:
        if token not in stage3: errs.append(f'Stage3 rolling contract token missing: {token}')
    core=(ROOT/'v4_realtime_core.py').read_text()
    if 'gpu_coord_channels: bool = False' not in core: errs.append('direct Stage1 core default is not CPU coordinate channels')
    report={'ok':not errs,'errors':errs,'stage1_per_slice':True,'stage2_per_slice':True,'stage3_only_multi_slice':True,'max_lookback_ft':450.0,'max_sequence_increments':9}
    print(json.dumps(report,indent=2))
    if errs: raise SystemExit('V4_PRODUCTION_SOURCE_CONTRACT_FAILED')
    print('V4_PRODUCTION_SOURCE_CONTRACT_OK')
if __name__=='__main__':main()
