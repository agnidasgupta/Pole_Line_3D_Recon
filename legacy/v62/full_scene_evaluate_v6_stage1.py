#!/usr/bin/env python3
"""Resumable exhaustive full-scene evaluation for the V6 Stage-1 dual-scale model."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import json
import os
import time
from contextlib import nullcontext

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from voxel_common import IGNORE_INDEX, load_npz_dense, read_json
from precision_common import (
    class_metrics_from_cm,
    cm_from_hist,
    fuse_scores,
    maybe_compile_model,
    search_thresholds,
    search_thresholds_class_specific,
    update_cm,
    update_score_hist,
    write_json_atomic,
)
from v6_common import SpanAwareGeoNet3D, build_v6_features, safe_id


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--split", choices=["val","test"], required=True)
    ap.add_argument("--calibration_json", default=None)
    ap.add_argument("--patch_size", type=int, default=64)
    ap.add_argument("--context_xy", type=int, default=256)
    ap.add_argument("--context_z", type=int, default=128)
    ap.add_argument("--core_size", type=int, default=48)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--build_workers", type=int, default=8)
    ap.add_argument("--base_channels", type=int, default=16)
    ap.add_argument("--use_coord_channels", type=int, default=1)
    ap.add_argument("--use_dist", type=int, default=1)
    ap.add_argument("--channels_last", type=int, default=1)
    ap.add_argument("--amp", choices=["none","bf16","fp16"], default="bf16")
    ap.add_argument("--compile_model", type=int, default=1)
    ap.add_argument("--compile_mode", default="reduce-overhead")
    ap.add_argument("--resume", type=int, default=1)
    ap.add_argument("--max_files", type=int, default=0)
    ap.add_argument("--score_bins", type=int, default=201)
    ap.add_argument("--score_sem_weight", type=float, default=.55)
    ap.add_argument("--score_binary_weight", type=float, default=.35)
    ap.add_argument("--score_object_weight", type=float, default=.10)
    ap.add_argument("--threshold_min", type=float, default=.01)
    ap.add_argument("--threshold_max", type=float, default=.99)
    ap.add_argument("--threshold_steps", type=int, default=99)
    ap.add_argument("--target_precision", type=float, default=.95)
    ap.add_argument("--target_recall", type=float, default=.95)
    ap.add_argument("--target_iou", type=float, default=.80)
    ap.add_argument("--pole_target_precision", type=float, default=.80)
    ap.add_argument("--pole_target_recall", type=float, default=.85)
    ap.add_argument("--pole_target_iou", type=float, default=.62)
    ap.add_argument("--line_target_precision", type=float, default=.58)
    ap.add_argument("--line_target_recall", type=float, default=.88)
    ap.add_argument("--line_target_iou", type=float, default=.52)
    ap.add_argument("--line_recall_weight", type=float, default=2.5)
    return ap.parse_args()


def setup_torch():
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=True
    torch.set_float32_matmul_precision("high")


def autocast_ctx(args):
    if args.amp == "none": return nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if args.amp=="bf16" else torch.float16)


def tile_centers(grid_size, core):
    gx,gy,gz=[int(v) for v in grid_size]
    return [np.array([x+core//2,y+core//2,z+core//2],dtype=np.int64)
            for z in range(0,gz,core) for y in range(0,gy,core) for x in range(0,gx,core)]


def load_model(args, in_ch):
    ckpt=torch.load(args.model_path,map_location="cpu",weights_only=False)
    m=SpanAwareGeoNet3D(in_ch=in_ch,base=args.base_channels)
    m.load_state_dict(ckpt["model_state"])
    m=m.cuda().eval()
    if args.channels_last: m=m.to(memory_format=torch.channels_last_3d)
    m,compiled=maybe_compile_model(m,bool(args.compile_model),args.compile_mode)
    print("torch.compile active:",compiled)
    return m


def build_one(item,center,grid_size,args):
    fine,coarse,labels,_,_,_,_=build_v6_features(
        item,center,tuple(grid_size),args.patch_size,args.context_xy,args.context_z,
        bool(args.use_coord_channels),bool(args.use_dist))
    return fine,coarse,labels


def process_file(rec,model,args,grid_size,centers,pool):
    item=load_npz_dense(rec["npz_path"],tuple(grid_size),use_dist=bool(args.use_dist))
    hist=np.zeros((3,args.score_bins,args.score_bins),dtype=np.int64)
    raw_cm=np.zeros((3,3),dtype=np.int64)
    pad=(args.patch_size-args.core_size)//2
    sl=slice(pad,pad+args.core_size)
    valid_voxels=0
    for start in range(0,len(centers),args.batch_size):
        bc=centers[start:start+args.batch_size]
        built=list(pool.map(lambda c: build_one(item,c,grid_size,args),bc))
        xf=torch.from_numpy(np.stack([b[0] for b in built])).cuda(non_blocking=True)
        xc=torch.from_numpy(np.stack([b[1] for b in built])).cuda(non_blocking=True)
        y=torch.from_numpy(np.stack([b[2] for b in built])).cuda(non_blocking=True)
        if args.channels_last:
            xf=xf.contiguous(memory_format=torch.channels_last_3d)
            xc=xc.contiguous(memory_format=torch.channels_last_3d)
        with torch.inference_mode(),autocast_ctx(args):
            out=model(xf,xc)
        yc=y[:,sl,sl,sl]
        sem=out["semantic"][:,:,sl,sl,sl]
        ps,ls=fuse_scores(out,args.score_sem_weight,args.score_binary_weight,args.score_object_weight)
        ps=ps[:,sl,sl,sl]; ls=ls[:,sl,sl,sl]
        update_cm(raw_cm,yc,sem.argmax(dim=1))
        update_score_hist(hist,yc,ps,ls,args.score_bins)
        valid_voxels += int((yc!=IGNORE_INDEX).sum().item())
    return hist,raw_cm,valid_voxels


def plot_cm(cm,path,title):
    cm=np.asarray(cm)
    fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(cm); fig.colorbar(im,ax=ax)
    names=["other","pole","powerline"]
    ax.set_xticks(range(3),names,rotation=25,ha="right"); ax.set_yticks(range(3),names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    for i in range(3):
        for j in range(3): ax.text(j,i,f"{int(cm[i,j]):,}",ha="center",va="center")
    fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)


def main():
    args=parse_args()
    if args.context_xy % args.patch_size != 0 or args.context_z % args.patch_size != 0:
        raise ValueError("context_xy/context_z must be divisible by patch_size")
    if args.patch_size<args.core_size or (args.patch_size-args.core_size)%2: raise ValueError("invalid patch/core")
    if args.split=="test" and not args.calibration_json: raise ValueError("test requires calibration_json")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    setup_torch(); os.makedirs(args.output_dir,exist_ok=True)
    done_dir=os.path.join(args.output_dir,"done"); os.makedirs(done_dir,exist_ok=True)
    summary=read_json(os.path.join(args.dataset_dir,"manifests","summary.json"))
    records=read_json(os.path.join(args.dataset_dir,"manifests",f"{args.split}.json"))
    if args.max_files>0: records=records[:args.max_files]
    grid_size=summary["grid_size_xyz"]
    in_ch=1+(3 if args.use_coord_channels else 0)+(1 if args.use_dist else 0)
    model=load_model(args,in_ch); centers=tile_centers(grid_size,args.core_size)
    aggregate_hist=np.zeros((3,args.score_bins,args.score_bins),dtype=np.int64)
    aggregate_raw=np.zeros((3,3),dtype=np.int64)
    total=0; rows=[]; started=time.perf_counter()
    with futures.ThreadPoolExecutor(max_workers=args.build_workers) as pool:
        for i,rec in enumerate(records,start=1):
            sid=safe_id(rec); dp=os.path.join(done_dir,sid+".npz")
            if args.resume and os.path.exists(dp):
                d=np.load(dp); hist=d["hist"]; raw=d["raw_cm"]; vox=int(d["valid_voxels"]); sec=float(d["elapsed_seconds"])
            else:
                t0=time.perf_counter(); hist,raw,vox=process_file(rec,model,args,grid_size,centers,pool); sec=time.perf_counter()-t0
                np.savez_compressed(dp,hist=hist,raw_cm=raw,valid_voxels=np.int64(vox),elapsed_seconds=np.float64(sec))
            aggregate_hist+=hist; aggregate_raw+=raw; total+=vox
            rows.append({"index":i,"id":sid,"group_id":rec.get("group_id",""),"valid_voxels":vox,"elapsed_seconds":sec})
            write_json_atomic({"split":args.split,"completed_files":i,"total_files":len(records),"percent":100*i/max(len(records),1),
                               "last_file":sid,"valid_voxels":total,"elapsed_seconds":time.perf_counter()-started},
                              os.path.join(args.output_dir,"progress.json"))
            print(f"[{args.split}] {i}/{len(records)} {sid} voxels={total:,}")
    raw_rows,raw_miou=class_metrics_from_cm(aggregate_raw)
    if args.split=="val":
        calibrated,search=search_thresholds_class_specific(
            aggregate_hist,args.threshold_min,args.threshold_max,args.threshold_steps,
            pole_target_precision=args.pole_target_precision,pole_target_recall=args.pole_target_recall,pole_target_iou=args.pole_target_iou,
            line_target_precision=args.line_target_precision,line_target_recall=args.line_target_recall,line_target_iou=args.line_target_iou,
            line_recall_weight=args.line_recall_weight)
        calibration={"pole_threshold":calibrated["pole_threshold"],"line_threshold":calibrated["line_threshold"],
                     "score_weights":{"semantic":args.score_sem_weight,"binary":args.score_binary_weight,"objectness":args.score_object_weight},
                     "validation_metrics":calibrated,
                     "pole_targets":{"precision":args.pole_target_precision,"recall":args.pole_target_recall,"iou":args.pole_target_iou},
                     "line_targets":{"precision":args.line_target_precision,"recall":args.line_target_recall,"iou":args.line_target_iou},
                     "selection_mode":"class_specific_line_recall_aware"}
        write_json_atomic(calibration,os.path.join(args.output_dir,"calibration.json"))
        simple=[{k:v for k,v in r.items() if k not in {"cm","class_metrics"}} for r in search]
        with open(os.path.join(args.output_dir,"threshold_search.csv"),"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(simple[0].keys())); w.writeheader(); w.writerows(simple)
    else:
        calibration=read_json(args.calibration_json); pt=float(calibration["pole_threshold"]); lt=float(calibration["line_threshold"])
        cm=cm_from_hist(aggregate_hist,pt,lt); cls,miou=class_metrics_from_cm(cm)
        calibrated={"pole_threshold":pt,"line_threshold":lt,"cm":cm.tolist(),"class_metrics":cls,"miou":miou,
                    "pole_precision":cls[1]["precision"],"pole_recall":cls[1]["recall"],"pole_iou":cls[1]["iou"],
                    "line_precision":cls[2]["precision"],"line_recall":cls[2]["recall"],"line_iou":cls[2]["iou"],
                    "all_targets_met":all([cls[1]["precision"]>=args.pole_target_precision,cls[1]["recall"]>=args.pole_target_recall,
                                             cls[1]["iou"]>=args.pole_target_iou,cls[2]["precision"]>=args.line_target_precision,
                                             cls[2]["recall"]>=args.line_target_recall,cls[2]["iou"]>=args.line_target_iou])}
    result={"split":args.split,"completed":True,"model_path":args.model_path,"files":len(records),"evaluated_occupied_voxels":total,
            "elapsed_seconds":time.perf_counter()-started,"raw_semantic_argmax":{"cm":aggregate_raw.tolist(),"class_metrics":raw_rows,"miou":raw_miou},
            "calibrated":calibrated,"config":vars(args),"note":"Every occupied voxel counted once through non-overlapping cores; each prediction uses native 64^3 plus anisotropic downsampled context."}
    write_json_atomic(result,os.path.join(args.output_dir,"full_scene_metrics.json"))
    with open(os.path.join(args.output_dir,"per_file_runtime.csv"),"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    plot_cm(aggregate_raw,os.path.join(args.output_dir,"confusion_matrix_raw.png"),f"V6 Stage-1 {args.split} raw")
    plot_cm(calibrated["cm"],os.path.join(args.output_dir,"confusion_matrix_calibrated.png"),f"V6 Stage-1 {args.split} calibrated")
    print(json.dumps({"split":args.split,"calibrated":{k:calibrated[k] for k in ["pole_precision","pole_recall","pole_iou","line_precision","line_recall","line_iou","all_targets_met"]}},indent=2))


if __name__=="__main__": main()
