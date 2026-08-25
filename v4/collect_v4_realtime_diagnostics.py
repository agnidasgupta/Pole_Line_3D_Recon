#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

TIMING_COLUMNS = [
    'csv_read_ms','sparse_item_prep_ms','stage1_total_ms','stage1_wall_ms',
    'patch_build_ms','host_batch_pack_ms','host_pin_ms','h2d_ms','sparse_h2d_cuda_ms',
    'gpu_sparse_scatter_ms','gpu_patch_extract_ms','gpu_feature_assembly_ms','gpu_model_ms',
    'gpu_gather_ms','d2h_gather_ms','stage1_artifact_load_ms','stage1_artifact_write_ms',
    'stage1_manifest_write_ms','stage2_component_ms','stage2_refiner_parametric_ms','stage2_total_ms',
    'stage2_artifact_write_ms','stage2_manifest_write_ms','stage3_incremental_ms','stage3_algorithm_ms',
    'stage3_fragment_join_ms','stage3_span_completion_pre_ms','stage3_hidden_pole_ms',
    'stage3_span_completion_post_ms','stage3_chain_build_attachment_ms','stage3_output_write_ms',
    'stage3_wrapper_overhead_ms','arrival_to_publish_ms','end_to_end_update_ms',
]


def stats(path, cols):
    p = Path(path)
    if not p.is_file():
        return {'missing': str(p)}
    try:
        d = pd.read_csv(p)
    except EmptyDataError:
        return {'path': str(p), 'rows': 0, 'empty': True}
    out = {'path': str(p), 'rows': len(d)}
    for c in cols:
        if c not in d.columns:
            continue
        a = pd.to_numeric(d[c], errors='coerce').dropna().to_numpy(float)
        if len(a):
            out[c] = {
                'mean': float(a.mean()), 'p50': float(np.quantile(a,.5)),
                'p95': float(np.quantile(a,.95)), 'max': float(a.max()),
            }
    return out


def readj(p):
    p = Path(p)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as exc:
        return {'read_error': repr(exc), 'path': str(p)}


def read_text(p):
    p = Path(p)
    if not p.is_file():
        return None
    try:
        return p.read_text()
    except Exception as exc:
        return f'<read_error {exc!r}>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_root', required=True)
    ap.add_argument('--replay_dir', default=None)
    ap.add_argument('--diagnostics_dir', default=None)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    r = Path(a.run_root)
    report = {
        'run_root': str(r),
        'stage2_training_completed': readj(r/'STAGE2_TRAINING_COMPLETED.json'),
        'stage2_refiner_metrics': readj(r/'stage2_refiner/local_refiner_metrics.json'),
        'stage2_mining_timing': stats(r/'stage2_mining/stage2_mining_runtime.csv', [
            'load_ms','patch_build_ms','h2d_ms','gpu_feature_assembly_ms','gpu_model_ms',
            'd2h_gather_ms','component_ms','total_ms']),
    }
    if a.diagnostics_dir:
        droot = Path(a.diagnostics_dir)
        report['diagnostics_dir'] = str(droot)
        report['runtime_variant_equivalence'] = readj(droot/'runtime_variant_equivalence.json')
        report['runtime_mode_env'] = read_text(droot/'v4_runtime_mode.env')
        report['batch_size_env'] = read_text(droot/'v4_batch_size.env')
        report['production_acceptance'] = read_text(droot/'PRODUCTION_ACCEPTANCE_OK.txt')
        report['gated_deployment_sha256'] = read_text(droot/'gated_deployment_sha256.txt')
        report['accepted_deployment_sha256'] = read_text(droot/'production_acceptance_deployment_sha256.txt')
        report['batch_size_summary'] = readj(droot/'batch_sweep/batch_size_summary.json')
    if a.replay_dir:
        q = Path(a.replay_dir)
        report['replay_dir'] = str(q)
        report['replay_completed'] = readj(q/'COMPLETED.json')
        report['replay_verification'] = readj(q/'REALTIME_REPLAY_VERIFICATION.json')
        report['realtime_timing'] = stats(q/'realtime_slice_timing.csv', TIMING_COLUMNS)
        report['stage1_manifest_rows'] = stats(q/'stage1_manifest.csv', [])
        report['stage2_manifest_rows'] = stats(q/'inference_manifest.csv', [])
        latest = q/'stage3_incremental'
        completed = list(latest.rglob('COMPLETED.json')) if latest.exists() else []
        report['incremental_stage3_snapshots'] = len(completed)
        errors = list((q/'errors').glob('*FAILED*.json')) if (q/'errors').exists() else []
        report['failure_artifacts'] = [str(x) for x in sorted(errors)]
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
