#!/usr/bin/env python3
"""Prepare V6.2 Stage-1 training data using ONLY slice-local columns.

World/center columns are intentionally ignored even when present.  The output
NPZ files contain only local x/y/z, labels, dist_center_ft, and an empty
hard-negative mask.  Splits are by geography/session group so slices from one
session never cross train/val/test boundaries. Missing slice sequence numbers
are normal and require no special handling here.
"""
from __future__ import annotations
import argparse, gzip, json, os, random, re, time
from pathlib import Path
import numpy as np
import pandas as pd

REQ=("label","x","y","z")
IGNORED={"center_x","center_y","center_z","world_x","world_y","world_z"}
SESSION_RE=re.compile(r"(session\d+)",re.I)


def args():
    p=argparse.ArgumentParser()
    p.add_argument("--input_dir",required=True)
    p.add_argument("--output_dir",required=True)
    p.add_argument("--val_frac",type=float,default=.15)
    p.add_argument("--test_frac",type=float,default=.15)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--min_grid_x",type=int,default=400)
    p.add_argument("--min_grid_y",type=int,default=400)
    p.add_argument("--min_grid_z",type=int,default=200)
    p.add_argument("--resume",type=int,default=1)
    return p.parse_args()


def safe_id(rel):
    s=str(rel).replace("\\","/")
    s=re.sub(r"\.csv(?:\.gz)?$","",s,flags=re.I)
    return re.sub(r"[^A-Za-z0-9_.-]+","__",s)


def group_id(rel):
    parts=Path(rel).parts
    top=parts[0] if parts else "ungrouped"
    second=parts[1] if len(parts)>1 else "session0"
    m=SESSION_RE.search(second)
    return f"{top}/{m.group(1).lower() if m else second}"


def read_csv(path):
    return pd.read_csv(path)


def split_by_groups(records,val_frac,test_frac,seed):
    groups=sorted({r["group_id"] for r in records}); rng=random.Random(seed); rng.shuffle(groups)
    if len(groups)<3: raise RuntimeError(f"Need >=3 geography/session groups, found {len(groups)}")
    nt=max(1,int(round(len(groups)*test_frac))); nv=max(1,int(round(len(groups)*val_frac)))
    if nt+nv>=len(groups): nt=nv=1
    tg=set(groups[:nt]); vg=set(groups[nt:nt+nv]); rg=set(groups[nt+nv:])
    out={"train":[r for r in records if r["group_id"] in rg],"val":[r for r in records if r["group_id"] in vg],"test":[r for r in records if r["group_id"] in tg]}
    if any(not v for v in out.values()): raise RuntimeError("Empty split generated")
    return out,{"train":sorted(rg),"val":sorted(vg),"test":sorted(tg)}


def write_json(obj,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(json.dumps(obj,indent=2))


def main():
    a=args(); root=Path(a.input_dir).resolve(); out=Path(a.output_dir); npz=out/"npz"; man=out/"manifests"
    npz.mkdir(parents=True,exist_ok=True); man.mkdir(parents=True,exist_ok=True)
    files=sorted(root.rglob("*.csv"))+sorted(root.rglob("*.csv.gz"))
    print(f"[prepare-v62] discovered {len(files)} CSV files")
    records=[]; skipped=[]; ignored_seen=set(); maxxyz=np.array([a.min_grid_x-1,a.min_grid_y-1,a.min_grid_z-1],int)
    for i,p in enumerate(files,1):
        rel=str(p.relative_to(root)); sid=safe_id(rel); op=npz/f"{sid}.npz"
        try:
            df=read_csv(p)
            ignored_seen.update(IGNORED.intersection(df.columns))
            miss=[c for c in REQ if c not in df.columns]
            if miss: raise ValueError(f"missing required local columns {miss}")
            # Explicitly select only the columns Stage 1 is allowed to see.
            cols=list(REQ)+(["dist_center_ft"] if "dist_center_ft" in df.columns else [])
            local=df.loc[:,cols].copy()
            for c in REQ: local[c]=pd.to_numeric(local[c],errors="coerce")
            if "dist_center_ft" in local: local["dist_center_ft"]=pd.to_numeric(local["dist_center_ft"],errors="coerce")
            good=np.isfinite(local[["label","x","y","z"]]).all(axis=1).to_numpy(); local=local.loc[good]
            if local.empty: raise ValueError("no valid local-coordinate rows")
            labels=local.label.astype(int).clip(0,2).to_numpy(np.int16)
            coords=np.rint(local[["x","y","z"]].to_numpy(float)).astype(np.int32)
            valid=(coords>=0).all(axis=1); coords=coords[valid]; labels=labels[valid]; local=local.iloc[np.flatnonzero(valid)]
            if len(coords)==0: raise ValueError("no nonnegative voxel coordinates")
            dist=(local["dist_center_ft"].fillna(0).to_numpy(np.float32) if "dist_center_ft" in local else np.zeros(len(coords),np.float32))
            hard=np.zeros(len(coords),np.uint8)
            if not (a.resume and op.exists()): np.savez_compressed(op,coords=coords,labels=labels,dist=dist,hardneg=hard)
            counts={str(c):int((labels==c).sum()) for c in (0,1,2)}
            rec={"id":sid,"source_csv":str(p),"source_relpath":rel,"group_id":group_id(rel),"npz_path":str(op),
                 "num_voxels":int(len(coords)),"label_counts":counts,"hardneg_count":0,
                 "min_xyz":coords.min(0).astype(int).tolist(),"max_xyz":coords.max(0).astype(int).tolist(),
                 "bucket":"both" if counts["1"] and counts["2"] else ("pole" if counts["1"] else ("line" if counts["2"] else "none"))}
            records.append(rec); maxxyz=np.maximum(maxxyz,coords.max(0))
            if i%100==0 or i==len(files): print(f"[prepare-v62] {i}/{len(files)} valid={len(records)} skipped={len(skipped)}")
        except Exception as e:
            skipped.append({"source_relpath":rel,"reason":str(e)}); print(f"[prepare-v62] SKIP {rel}: {e}")
    if not records: raise RuntimeError("No valid training files")
    splits,split_groups=split_by_groups(records,a.val_frac,a.test_frac,a.seed)
    for k,v in splits.items(): write_json(v,man/f"{k}.json")
    write_json(records,man/"all.json")
    summary={"created_at":time.strftime("%F %T"),"input_dir":str(root),"num_files":len(records),"skipped":skipped,
             "grid_size_xyz":(maxxyz+1).astype(int).tolist(),"split_sizes":{k:len(v) for k,v in splits.items()},
             "split_groups":split_groups,"groups":sorted({r['group_id'] for r in records}),
             "total_label_counts":{str(c):int(sum(r['label_counts'][str(c)] for r in records)) for c in (0,1,2)},
             "stage1_allowed_columns":["label","x","y","z","dist_center_ft"],
             "explicitly_ignored_columns":sorted(ignored_seen),
             "note":"center_* and world_* are never written to Stage-1 NPZs and cannot influence training."}
    write_json(summary,man/"summary.json"); print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
