#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--summary_json',required=True); p.add_argument('--output_env',required=True); a=p.parse_args()
    rows=json.load(open(a.summary_json))
    passing=[r for r in rows if bool(r.get('pass_equivalence'))]
    if not passing: raise SystemExit('No numerically equivalent batch size passed')
    best=min(passing,key=lambda r:float(r['stage1_wall']['mean_ms']))
    out=Path(a.output_env); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(f"export V4_BATCH_SIZE={int(best['batch_size'])}\nexport V4_BATCH_SWEEP_JSON={Path(a.summary_json).resolve()}\n")
    print(json.dumps({'selected_batch_size':int(best['batch_size']),'mean_ms':best['stage1_wall']['mean_ms'],'p95_ms':best['stage1_wall']['p95_ms'],'passing_batch_sizes':[int(r['batch_size']) for r in passing]},indent=2))
if __name__=='__main__':main()
