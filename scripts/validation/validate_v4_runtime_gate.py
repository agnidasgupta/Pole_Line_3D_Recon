#!/usr/bin/env python3
from __future__ import annotations
import argparse,json

def main():
    p=argparse.ArgumentParser();p.add_argument('--gate_json',required=True);a=p.parse_args();d=json.load(open(a.gate_json));s=d.get('summary',{});fg=s.get('full_gpu',{})
    if not fg.get('pass_recommended',False):raise SystemExit(f"FULL_GPU_EQUIVALENCE_FAILED: {fg}")
    print(f"V4_RUNTIME_GATE_OK full_gpu=pass recommended={d.get('recommended_runtime')} active_gpu_pass={bool(s.get('active_gpu',{}).get('pass_recommended',False))}")
if __name__=='__main__':main()
