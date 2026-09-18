"""Synthetic V4 A/B benchmark; never a production quality or latency claim.

Uses the real model and sparse inference code, identical seeded random weights
for both paths, mock calibration, and synthetic occupied voxels. Does not create
or overwrite a checkpoint/calibration file. Stage 2 geometry is tested separately
with test_support_checks.py; random network predictions are not meaningful poles.
"""
import argparse
import importlib.util
import json
import os
from contextlib import nullcontext
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'v4'), str(ROOT / 'ops/v4_stage1_inference_opt2')]
import v4_realtime_core as candidate_core
from voxel_common import MultiHeadVoxelNet3D


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def synthetic_item(cores, seed):
    rng = np.random.default_rng(seed)
    # Deterministic cores include grid boundaries. Retain row sorting/deduplication
    # from the production parser, not a hand-built approximation of its item.
    keys = np.array([(x, y, z) for z in range(5) for y in range(9) for x in range(9)])
    chosen = keys[np.linspace(0, len(keys) - 1, cores, dtype=int)]
    points = []
    for key in chosen:
        low = key * 48
        high = np.minimum(low + 48, [400, 400, 200])
        points.append(rng.integers(low, high, size=(1800, 3)))
    coords = np.vstack(points)
    df = pd.DataFrame(coords, columns=['x', 'y', 'z'])
    df['dist_center_ft'] = np.linalg.norm(coords - [200, 200, 100], axis=1) * .5
    return candidate_core.build_sparse_item_from_dataframe(df, (400, 400, 200))


def nvtx_pre_hook(label):
    def enter(module, inputs):
        torch.cuda.nvtx.range_push(label)
        # Hooks must return None: NVTX itself returns an integer range depth.
    return enter


def nvtx_post_hook(module, inputs, output):
    torch.cuda.nvtx.range_pop()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline-root', type=Path, required=True)
    p.add_argument('--cores', type=int, nargs='+', default=[1, 12, 13, 120])
    p.add_argument('--iterations', type=int, default=5)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--validate-inputs-only', action='store_true')
    p.add_argument('--nsys', action='store_true', help='NVTX layers/phases and CUDA profiler capture after warmup; one core-count case only')
    p.add_argument('--output', default='artifacts/layout_validation/stage1_h100.json')
    args = p.parse_args()
    if args.nsys:
        if len(args.cores) != 1:
            p.error('--nsys requires exactly one --cores value')
        os.environ['POLELINE_NVTX'] = '1'
    if args.iterations < 1 or args.warmup < 0 or any(k < 1 or k > 405 for k in args.cores):
        p.error('cores must be in 1..405, iterations >= 1, warmup >= 0')
    items = [(k, synthetic_item(k, args.seed + k)) for k in args.cores]
    for k, item in items:
        actual = len(candidate_core.active_core_groups(item['coords']))
        assert actual == k, (actual, k)
    if args.validate_inputs_only:
        print(json.dumps([{'active_cores': k, 'rows': len(item['coords'])} for k, item in items]))
        return
    if not torch.cuda.is_available():
        raise SystemExit('CUDA GPU required; no CPU timing substituted')
    device_name = torch.cuda.get_device_name(0)
    if 'H100' not in device_name:
        raise SystemExit(f'H100 required for this deployment benchmark; found {device_name}')
    candidate_core.setup_torch()
    torch.manual_seed(args.seed)
    model = MultiHeadVoxelNet3D(in_ch=5, base=16, emb_dim=8).eval().cuda()
    model = model.to(memory_format=torch.channels_last_3d)
    assert sum(x.numel() for x in model.parameters()) == 341518
    config = {'patch_size': 64, 'use_coord_channels': 1, 'use_dist': 1}
    calibration = {'score_sem_weight': .55, 'score_binary_weight': .35,
                   'score_object_weight': .1, 'pole_threshold': .5, 'line_threshold': .5}
    path = ROOT / 'ops/v4_stage1_inference_opt2/v4_realtime_core_opt2.py'
    baseline = load('mock_baseline_opt2', args.baseline_root / 'ops/v4_stage1_inference_opt2/v4_realtime_core_opt2.py')
    baseline.reference = load('mock_original_core', args.baseline_root / 'v4/v4_realtime_core.py')
    candidate = load('mock_candidate_opt2', path)
    modules = {'baseline': baseline, 'candidate': candidate}
    reports = []
    for cores, item in items:
        workspaces = {name: module.V4SparseGpuWorkspace() for name, module in modules.items()}
        def infer(name):
            return modules[name].predict_v4_sparse_rows_opt(
                item, model, config, calibration, batch_size=12, amp='bf16',
                workspace=workspaces[name], detailed_cuda_timing=True,
                cache_coordinate_channels=False)
        for _ in range(args.warmup):
            infer('baseline'); infer('candidate')
        hooks = []
        if args.nsys:
            for layer_name, layer in model.named_modules():
                label = 'layer/' + (layer_name or 'MultiHeadVoxelNet3D') + ':' + type(layer).__name__
                hooks.append(layer.register_forward_pre_hook(nvtx_pre_hook(label)))
                hooks.append(layer.register_forward_hook(nvtx_post_hook, always_call=True))
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
        times = {name: [] for name in modules}
        failures = []
        baseline_stability = None
        for iteration in range(args.iterations):
            outputs = {}
            order = ['baseline', 'candidate'] if iteration % 2 == 0 else ['candidate', 'baseline']
            for name in order:
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.cuda.nvtx.range(f'slice/{name}/iteration_{iteration}') if args.nsys else nullcontext():
                    outputs[name] = infer(name)
                    torch.cuda.synchronize()
                times[name].append((time.perf_counter() - start) * 1000)
            for key in ('pole', 'line', 'semantic', 'objectness'):
                a, b = outputs['baseline'][key], outputs['candidate'][key]
                if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                    failures.append(f'iteration {iteration}: {key} differs')
                if baseline_stability is not None and a.tobytes() != baseline_stability[key]:
                    failures.append(f'iteration {iteration}: baseline {key} is unstable')
            baseline_stability = {k: outputs['baseline'][k].tobytes() for k in ('pole', 'line', 'semantic', 'objectness')}
            old_labels = candidate_core.label_from_scores(outputs['baseline']['pole'], outputs['baseline']['line'], .5, .5)
            new_labels = candidate_core.label_from_scores(outputs['candidate']['pole'], outputs['candidate']['line'], .5, .5)
            if not np.array_equal(old_labels, new_labels):
                failures.append(f'iteration {iteration}: mock labels differ')
        if args.nsys:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            for hook in hooks:
                hook.remove()
        report = {'active_cores': cores, 'occupied_rows': len(item['coords']),
                  'exact_and_stable': not failures, 'failures': failures,
                  'timings_ms': {name: {'mean': float(np.mean(t)), 'p50': float(np.median(t)),
                                       'p95': float(np.percentile(t, 95)), 'samples': t} for name, t in times.items()}}
        reports.append(report)
        print(json.dumps(report), flush=True)
        if failures:
            break
    result = {'synthetic': True, 'production_equivalence_verified': False,
              'nsys_instrumented': args.nsys,
              'scope': 'Stage1 sparse inference only; excludes CSV IO, Stage2 and writes',
              'device': device_name, 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'cudnn': torch.backends.cudnn.version(), 'python': platform.python_version(),
              'seed': args.seed, 'checkpoint': 'none: seeded random weights',
              'calibration': 'mock coefficients and thresholds', 'warmup': args.warmup,
              'batch': 12, 'patch': 64, 'core': 48, 'amp': 'bf16', 'reports': reports}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    raise SystemExit(0 if all(r['exact_and_stable'] for r in reports) else 1)


if __name__ == '__main__':
    main()
