#!/usr/bin/env python3
"""Compare Stage-3 reconstruction/audit CSV snapshots between two runs."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

NAMES = {
    "world_poles.csv",
    "conductor_chains.csv",
    "conductor_vertices.csv",
    "spans.csv",
    "accepted_fragment_joins.csv",
    "inferred_hidden_poles.csv",
    "span_completion_paths.csv",
    "prevented_polygon_connections.csv",
}


def pa():
    p=argparse.ArgumentParser()
    p.add_argument("baseline")
    p.add_argument("experiment")
    p.add_argument("--atol",type=float,default=1e-9)
    return p.parse_args()


def csv_map(root):
    root=Path(root).resolve(); out={}
    for p in root.rglob("*.csv"):
        if p.name in NAMES:
            out[str(p.relative_to(root))]=p
    return out


def same_df(a,b,atol):
    if list(a.columns)!=list(b.columns) or a.shape!=b.shape:return False,"shape/columns"
    for c in a.columns:
        aa=a[c]; bb=b[c]
        if pd.api.types.is_numeric_dtype(aa) and pd.api.types.is_numeric_dtype(bb):
            x=pd.to_numeric(aa,errors="coerce").to_numpy(float); y=pd.to_numeric(bb,errors="coerce").to_numpy(float)
            if not np.allclose(x,y,rtol=0.0,atol=atol,equal_nan=True):return False,f"numeric:{c}"
        else:
            x=aa.fillna("<NA>").astype(str).to_numpy(); y=bb.fillna("<NA>").astype(str).to_numpy()
            if not np.array_equal(x,y):return False,f"text:{c}"
    return True,""


def main():
    a=pa(); A=csv_map(a.baseline); B=csv_map(a.experiment)
    onlyA=sorted(set(A)-set(B)); onlyB=sorted(set(B)-set(A)); failures=[]
    if onlyA: failures.append(("missing_experiment",onlyA[:20]))
    if onlyB: failures.append(("extra_experiment",onlyB[:20]))
    checked=0
    for rel in sorted(set(A)&set(B)):
        da=pd.read_csv(A[rel]); db=pd.read_csv(B[rel]); ok,why=same_df(da,db,a.atol); checked+=1
        if not ok: failures.append((rel,why))
    print(f"checked_csvs={checked} baseline_files={len(A)} experiment_files={len(B)}")
    if failures:
        print("STAGE3_OUTPUT_EQUIVALENCE_FAIL")
        for x in failures[:50]: print(x)
        return 1
    print("STAGE3_OUTPUT_EQUIVALENCE_OK")
    return 0
if __name__=="__main__":raise SystemExit(main())
