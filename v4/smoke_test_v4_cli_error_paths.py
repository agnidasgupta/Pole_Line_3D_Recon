#!/usr/bin/env python3
"""Exercise CLI failure paths that must fail clearly before expensive GPU work."""
from __future__ import annotations
import subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = sys.executable


def expect_fail(args, needles):
    p = subprocess.run([PYTHON, *map(str,args)], cwd=HERE, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode == 0:
        raise AssertionError(f"Expected failure but command passed: {' '.join(map(str,args))}\n{p.stdout}")
    text = p.stdout.lower()
    if needles and not any(n.lower() in text for n in needles):
        raise AssertionError(f"Failure message missing {needles}: {' '.join(map(str,args))}\n{text[-4000:]}")


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        missing = root/'does_not_exist'
        out = root/'out'
        # Stage 1 must reject a nonexistent/missing session before model loading.
        expect_fail(['run_v4_stage1.py','--input_dir',root,'--session_filter','Missing/session1','--output_dir',out,
                     '--model_path',missing/'m.pt','--calibration_json',missing/'c.json'], ['filenotfounderror','missing'])
        # Stage 2/3 must reject absent durable upstream manifests explicitly.
        expect_fail(['run_v4_stage2.py','--stage1_dir',missing,'--output_dir',out,'--session_filter','Geo/session1',
                     '--stage2_bundle',missing/'s.joblib'], ['stage1_manifest.csv','filenotfounderror'])
        expect_fail(['run_v4_stage3.py','--stage2_dir',missing,'--output_dir',out,'--session_filter','Geo/session1'],
                    ['inference_manifest.csv','filenotfounderror'])
        expect_fail(['reconstruct_v4_stage3.py','--inference_dir',missing,'--output_dir',out,'--session_filter','Geo/session1',
                     '--latest_slice','1','--max_sequence_gap','9','--slice_length_ft','50','--max_span_length_ft','450'],
                    ['inference manifest','filenotfounderror'])
        expect_fail(['summarize_v4_realtime_timing.py','--timing_csv',missing/'timing.csv'], ['timing csv not found'])
        expect_fail(['verify_v4_realtime_replay.py','--replay_dir',missing], ['error missing'])
        expect_fail(['select_v4_runtime_mode.py','--gate_json',missing/'gate.json','--output_env',out/'mode.env'], ['filenotfounderror'])
        expect_fail(['select_v4_batch_size.py','--summary_json',missing/'summary.json','--output_env',out/'batch.env'], ['filenotfounderror'])
        # Invalid physical contract must be rejected independently of filesystem contents.
        expect_fail(['run_v4_stage3.py','--stage2_dir',missing,'--output_dir',out,'--session_filter','Geo/session1',
                     '--max_sequence_gap','8','--slice_length_ft','50','--max_span_length_ft','450'], ['production stage3 contract'])
        expect_fail(['verify_v4_realtime_replay.py','--replay_dir',missing,'--max_sequence_gap','8'], ['production contract'])
    print('V4_CLI_ERROR_PATHS_OK missing_input=1 missing_stage_manifests=1 missing_timing=1 invalid_450ft_contract=1')


if __name__ == '__main__':
    main()
