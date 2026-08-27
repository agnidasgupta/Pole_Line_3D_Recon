#!/usr/bin/env python3
"""Summarize Stage-3 baseline vs incremental timing CSVs."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

def main():
 p=argparse.ArgumentParser();p.add_argument('baseline');p.add_argument('experiment');a=p.parse_args()
 A=pd.read_csv(a.baseline);B=pd.read_csv(a.experiment)
 keys=['stage3_fragment_join_ms','stage3_elapsed_ms','stage3_wall_ms','stage3_chain_build_attachment_ms','stage3_span_completion_pre_ms']
 print('metric,run,mean_ms,p50_ms,p95_ms,max_ms')
 for k in keys:
  for name,d in [('baseline',A),('experiment',B)]:
   if k not in d:continue
   x=pd.to_numeric(d[k],errors='coerce').dropna().to_numpy(float);x=x[x>0]
   if len(x):print(f'{k},{name},{x.mean():.3f},{np.quantile(x,.5):.3f},{np.quantile(x,.95):.3f},{x.max():.3f}')
  if k in A and k in B:
   x=pd.to_numeric(A[k],errors='coerce').dropna().to_numpy(float);y=pd.to_numeric(B[k],errors='coerce').dropna().to_numpy(float);x=x[x>0];y=y[y>0]
   if len(x) and len(y):print(f'{k},mean_speedup,{x.mean()/y.mean():.4f}x,,,')
 return 0
if __name__=='__main__':raise SystemExit(main())
