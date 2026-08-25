#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--gate_json',required=True)
    ap.add_argument('--output_env',required=True)
    a=ap.parse_args()
    d=json.load(open(a.gate_json))
    active=d.get('summary',{}).get('active_cpu',{})
    ok=bool(active.get('pass_recommended',False))
    if ok:
        mode='active_cpu'; all_cores=0; gpu_coords=0
    else:
        mode='full_cpu'; all_cores=1; gpu_coords=0
    out=Path(a.output_env); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(
        f'export V4_RUNTIME_MODE={mode}\n'
        f'export EVALUATE_ALL_CORES={all_cores}\n'
        f'export GPU_COORD_CHANNELS={gpu_coords}\n'
        f'export V4_RUNTIME_GATE_JSON={Path(a.gate_json).resolve()}\n'
    )
    print(json.dumps({'selected':mode,'evaluate_all_cores':all_cores,'gpu_coord_channels':gpu_coords,'active_cpu_gate':active},indent=2))
if __name__=='__main__':main()
