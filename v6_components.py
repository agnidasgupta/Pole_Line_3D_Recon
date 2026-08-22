#!/usr/bin/env python3
"""Component extraction, physical features, and refiner utilities for V6."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

FEATURE_COLUMNS = [
    "n_voxels", "score_mean", "score_std", "score_p10", "score_p50", "score_p90", "score_max",
    "vertical_head_mean", "horizontal_head_mean", "x_span_ft", "y_span_ft", "z_span_ft",
    "horizontal_span_ft", "bbox_density", "center_z_ft", "min_z_ft", "max_z_ft",
    "principal_dx", "principal_dy", "principal_dz", "principal_verticality",
    "linearity", "planarity", "scattering", "radius_p50_ft", "radius_p90_ft",
    "xy_path_length_ft", "xy_endpoint_distance_ft", "xy_tortuosity",
    "quadratic_rmse_ft", "sag_estimate_ft", "endpoint1_pole_dist_ft", "endpoint2_pole_dist_ft",
    "distinct_pole_supports", "attached_line_count", "exact_gt_fraction", "near_gt_fraction",
    "world_repeat_files", "world_repeat_sessions", "world_attached_poles",
    "world_pole_separation_ft", "world_span_candidate", "world_attached_span_count",
    "gt_disagreement_world_supported",
]


def atomic_json(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _finite_points(points):
    p = np.asarray(points, dtype=np.float64)
    return p[np.isfinite(p).all(axis=1)]


def pca_geometry(points_xyz: np.ndarray, voxel_size=.5):
    p = _finite_points(points_xyz)
    if len(p) < 2:
        return {
            "principal": np.array([0.,0.,1.]), "evals": np.zeros(3), "linearity":0., "planarity":0., "scattering":0.,
            "radius_p50_ft":0., "radius_p90_ft":0., "endpoints": np.vstack([p[0] if len(p) else np.zeros(3)]*2),
            "xy_path_length_ft":0., "xy_endpoint_distance_ft":0., "xy_tortuosity":1., "quadratic_rmse_ft":0., "sag_estimate_ft":0.,
        }
    c = np.median(p, axis=0)
    centered = p - c
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]
        direction /= max(np.linalg.norm(direction), 1e-9)
    except np.linalg.LinAlgError:
        direction = np.array([0.,0.,1.])
    cov = np.cov(centered.T) if len(p) > 2 else np.zeros((3,3))
    try:
        evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    except np.linalg.LinAlgError:
        evals = np.zeros(3)
    e1,e2,e3 = [float(max(v,0.)) for v in evals]
    linearity = (e1-e2)/max(e1,1e-9)
    planarity = (e2-e3)/max(e1,1e-9)
    scattering = e3/max(e1,1e-9)
    t = centered @ direction
    fitted = c + np.outer(t, direction)
    radial = np.linalg.norm(p - fitted, axis=1) * voxel_size
    endpoints = np.vstack([p[np.argmin(t)], p[np.argmax(t)]])

    # Horizontal ordering for lines. Z is never allowed to define progression.
    xy = p[:,:2]
    cxy = np.median(xy, axis=0)
    try:
        _, _, vhxy = np.linalg.svd(xy-cxy, full_matrices=False)
        dxy = vhxy[0]
    except np.linalg.LinAlgError:
        dxy = np.array([1.,0.])
    dxy = dxy / max(np.linalg.norm(dxy),1e-9)
    s = (xy-cxy) @ dxy
    order = np.argsort(s)
    op = p[order]
    endpoints = np.vstack([op[0], op[-1]])
    dif = np.diff(op[:,:2], axis=0)
    path = float(np.linalg.norm(dif,axis=1).sum()*voxel_size) if len(op)>1 else 0.
    endpoint_d = float(np.linalg.norm(op[-1,:2]-op[0,:2])*voxel_size) if len(op)>1 else 0.
    tort = path/max(endpoint_d,1e-6)
    rmse=sag=0.
    # Stable diagnostic sag fit.  np.polyfit can emit RankWarning for nearly
    # vertical/tiny components; those objects do not contain enough independent
    # horizontal support for a meaningful quadratic sag estimate anyway.
    if len(p)>=6:
        sf=np.asarray(s,dtype=np.float64)*float(voxel_size)
        zf=np.asarray(p[:,2],dtype=np.float64)*float(voxel_size)
        span=float(np.ptp(sf))
        if np.isfinite(sf).all() and np.isfinite(zf).all() and span>=max(float(voxel_size)*2.0,1e-6) and np.unique(np.round(sf,6)).size>=3:
            try:
                mid=0.5*(float(sf.min())+float(sf.max())); half=max(0.5*span,1e-9)
                u=(sf-mid)/half
                A=np.column_stack([u*u,u,np.ones_like(u)])
                cond=float(np.linalg.cond(A))
                if np.isfinite(cond) and cond<1e8:
                    coef,_,_,_=np.linalg.lstsq(A,zf,rcond=None)
                    fit=A@coef
                    rmse=float(np.sqrt(np.mean((fit-zf)**2)))
                    zend1=float(coef[0]-coef[1]+coef[2]); zend2=float(coef[0]+coef[1]+coef[2])
                    zmid=float(coef[2]); sag=float(max(0.,0.5*(zend1+zend2)-zmid))
            except (np.linalg.LinAlgError,ValueError,FloatingPointError):
                pass
    return {"principal":direction,"evals":evals,"linearity":linearity,"planarity":planarity,"scattering":scattering,
            "radius_p50_ft":float(np.quantile(radial,.5)) if len(radial) else 0.,
            "radius_p90_ft":float(np.quantile(radial,.9)) if len(radial) else 0.,"endpoints":endpoints,
            "xy_path_length_ft":path,"xy_endpoint_distance_ft":endpoint_d,"xy_tortuosity":tort,
            "quadratic_rmse_ft":rmse,"sag_estimate_ft":sag}


def _component_rows(mask, score, vertical, horizontal, class_name, voxel_size=.5, gt_labels=None, gt_distance=None, min_voxels=3):
    structure=np.ones((3,3,3),dtype=np.uint8)
    lab,n=ndimage.label(mask,structure=structure)
    objects=ndimage.find_objects(lab)
    rows=[]; points_by_id={}
    for cid,slc in enumerate(objects, start=1):
        if slc is None: continue
        local=lab[slc]==cid
        count=int(local.sum())
        if count<min_voxels: continue
        zz,yy,xx=np.nonzero(local)
        z0,y0,x0=slc[0].start,slc[1].start,slc[2].start
        pts=np.column_stack([xx+x0,yy+y0,zz+z0]).astype(np.int32)
        vals=score[pts[:,2],pts[:,1],pts[:,0]].astype(np.float32)
        vvals=vertical[pts[:,2],pts[:,1],pts[:,0]].astype(np.float32)
        hvals=horizontal[pts[:,2],pts[:,1],pts[:,0]].astype(np.float32)
        geo=pca_geometry(pts,voxel_size)
        mins=pts.min(axis=0); maxs=pts.max(axis=0); spans=(maxs-mins+1)*voxel_size
        horizontal_span=float(np.linalg.norm(spans[:2]))
        bbox_vol=float(np.prod(maxs-mins+1))
        exact=near=0.
        if gt_labels is not None:
            target_class=1 if class_name=="pole" else 2
            exact=float(np.mean(gt_labels[pts[:,2],pts[:,1],pts[:,0]]==target_class))
        if gt_distance is not None:
            near=float(np.mean(gt_distance[pts[:,2],pts[:,1],pts[:,0]]<=2.5))
        pid=f"{class_name[0].upper()}{cid:05d}"
        d=geo["principal"]
        row={
            "component_id":pid,"class_name":class_name,"n_voxels":count,
            "score_mean":float(vals.mean()),"score_std":float(vals.std()),"score_p10":float(np.quantile(vals,.1)),
            "score_p50":float(np.quantile(vals,.5)),"score_p90":float(np.quantile(vals,.9)),"score_max":float(vals.max()),
            "vertical_head_mean":float(vvals.mean()),"horizontal_head_mean":float(hvals.mean()),
            "x_span_ft":float(spans[0]),"y_span_ft":float(spans[1]),"z_span_ft":float(spans[2]),
            "horizontal_span_ft":horizontal_span,"bbox_density":float(count/max(bbox_vol,1.)),
            "center_x":float(pts[:,0].mean()),"center_y":float(pts[:,1].mean()),"center_z":float(pts[:,2].mean()),
            "center_z_ft":float(pts[:,2].mean()*voxel_size),"min_z_ft":float(mins[2]*voxel_size),"max_z_ft":float(maxs[2]*voxel_size),
            "principal_dx":float(d[0]),"principal_dy":float(d[1]),"principal_dz":float(d[2]),
            "principal_verticality":float(abs(d[2])),"linearity":float(geo["linearity"]),"planarity":float(geo["planarity"]),
            "scattering":float(geo["scattering"]),"radius_p50_ft":geo["radius_p50_ft"],"radius_p90_ft":geo["radius_p90_ft"],
            "xy_path_length_ft":geo["xy_path_length_ft"],"xy_endpoint_distance_ft":geo["xy_endpoint_distance_ft"],
            "xy_tortuosity":geo["xy_tortuosity"],"quadratic_rmse_ft":geo["quadratic_rmse_ft"],"sag_estimate_ft":geo["sag_estimate_ft"],
            "endpoint1_x":float(geo["endpoints"][0,0]),"endpoint1_y":float(geo["endpoints"][0,1]),"endpoint1_z":float(geo["endpoints"][0,2]),
            "endpoint2_x":float(geo["endpoints"][1,0]),"endpoint2_y":float(geo["endpoints"][1,1]),"endpoint2_z":float(geo["endpoints"][1,2]),
            "endpoint1_pole_dist_ft":999.,"endpoint2_pole_dist_ft":999.,"distinct_pole_supports":0.,"attached_line_count":0.,
            "exact_gt_fraction":exact,"near_gt_fraction":near,
        }
        rows.append(row); points_by_id[pid]=pts
    return pd.DataFrame(rows),lab,points_by_id


def add_attachment_features(poles: pd.DataFrame, lines: pd.DataFrame, voxel_size=.5, attachment_radius_ft=12.):
    if poles.empty or lines.empty:
        return poles,lines
    pole_points=np.column_stack([poles.center_x,poles.center_y,poles.max_z_ft/voxel_size])
    tree=cKDTree(pole_points*voxel_size)
    attached={cid:0 for cid in poles.component_id}
    for i,row in lines.iterrows():
        ends=np.array([[row.endpoint1_x,row.endpoint1_y,row.endpoint1_z],[row.endpoint2_x,row.endpoint2_y,row.endpoint2_z]])*voxel_size
        d,idx=tree.query(ends,k=1)
        lines.at[i,"endpoint1_pole_dist_ft"]=float(d[0]); lines.at[i,"endpoint2_pole_dist_ft"]=float(d[1])
        distinct=int(d[0]<=attachment_radius_ft and d[1]<=attachment_radius_ft and int(idx[0])!=int(idx[1]))
        lines.at[i,"distinct_pole_supports"]=2. if distinct else float((d<=attachment_radius_ft).sum())
        for dd,j in zip(d,idx):
            if dd<=attachment_radius_ft:
                attached[poles.iloc[int(j)].component_id]+=1
    for i,row in poles.iterrows():
        poles.at[i,"attached_line_count"]=float(attached.get(row.component_id,0))
    return poles,lines


def assign_training_targets(df: pd.DataFrame, positive_near=.50, negative_near=.02):
    if df.empty:
        df["target"]=[]; return df
    target=np.full(len(df),-1,dtype=np.int8)
    near=df.near_gt_fraction.to_numpy(float); exact=df.exact_gt_fraction.to_numpy(float)
    target[(near>=positive_near)|(exact>=.25)]=1
    target[(near<=negative_near)&(exact<=.01)]=0
    df=df.copy(); df["target"]=target
    return df


def _hysteresis_mask(occmask, primary, competitor, strong_threshold, weak_threshold, competition_ratio=.55):
    """Keep weak candidate voxels only when connected to a strong seed.

    This is deliberately analogous to edge hysteresis: a partially occluded wire
    may contain low-score voxels between confident line detections, but isolated
    weak responses do not become components by themselves.
    """
    strong_threshold=float(strong_threshold); weak_threshold=float(weak_threshold)
    if weak_threshold > strong_threshold:
        raise ValueError("weak_threshold must be <= strong_threshold")
    ratio=float(competition_ratio)
    weak=occmask & (primary>=weak_threshold) & (primary>=competitor*ratio)
    strong=occmask & (primary>=strong_threshold) & (primary>=competitor*ratio)
    lab,n=ndimage.label(weak,structure=np.ones((3,3,3),dtype=np.uint8))
    if n==0 or not np.any(strong):
        return np.zeros_like(occmask,dtype=bool)
    keep=np.unique(lab[strong]); keep=keep[keep>0]
    if len(keep)==0:
        return np.zeros_like(occmask,dtype=bool)
    return np.isin(lab,keep)


def extract_all_components(
    occ, scores: Dict[str,np.ndarray], pole_threshold=.18, line_threshold=.08, voxel_size=.5,
    gt_labels=None, pole_min_voxels=4, line_min_voxels=3, attachment_radius_ft=12.,
    line_weak_threshold=.04, line_competition_ratio=.55,
):
    occmask=occ>0
    pmask=occmask & (scores["pole"]>=pole_threshold) & (scores["pole"]>=scores["line"]*.8)
    lmask=_hysteresis_mask(
        occmask,scores["line"],scores["pole"],line_threshold,line_weak_threshold,
        competition_ratio=line_competition_ratio)
    pole_dist=line_dist=None
    if gt_labels is not None:
        pole_dist=ndimage.distance_transform_edt(gt_labels!=1) if np.any(gt_labels==1) else np.full(gt_labels.shape,np.inf,np.float32)
        line_dist=ndimage.distance_transform_edt(gt_labels!=2) if np.any(gt_labels==2) else np.full(gt_labels.shape,np.inf,np.float32)
    poles,plabel,ppoints=_component_rows(pmask,scores["pole"],scores["verticality"],scores["horizontality"],"pole",voxel_size,gt_labels,pole_dist,pole_min_voxels)
    lines,llabel,lpoints=_component_rows(lmask,scores["line"],scores["verticality"],scores["horizontality"],"line",voxel_size,gt_labels,line_dist,line_min_voxels)
    poles,lines=add_attachment_features(poles,lines,voxel_size,attachment_radius_ft)
    poles=assign_training_targets(poles,.45,.02); lines=assign_training_targets(lines,.35,.02)
    return {"poles":poles,"lines":lines,"pole_labels":plabel,"line_labels":llabel,"pole_points":ppoints,"line_points":lpoints}


def pole_physical_ok(row, min_height_ft=12., max_height_ft=85., min_radius_ft=.15, max_radius_ft=3.0,
                     min_verticality=.70, max_tilt_ratio=.50):
    h=float(row.z_span_ft); r=float(row.radius_p90_ft); hs=float(row.horizontal_span_ft)
    return bool(min_height_ft<=h<=max_height_ft and min_radius_ft<=r<=max_radius_ft
                and float(row.principal_verticality)>=min_verticality and hs/max(h,1e-6)<=max_tilt_ratio)


def line_physical_ok(row, min_horizontal_span_ft=8., max_verticality=.55, max_vertical_horizontal=.75,
                     max_tortuosity=1.8):
    h=float(row.horizontal_span_ft); z=float(row.z_span_ft)
    return bool(h>=min_horizontal_span_ft and float(row.principal_verticality)<=max_verticality
                and z/max(h,1e-6)<=max_vertical_horizontal and float(row.xy_tortuosity)<=max_tortuosity)


def apply_refiners(poles,lines,pole_model,line_model,pole_threshold,line_threshold,physical_cfg):
    poles=poles.copy(); lines=lines.copy()
    for df,model,thr,name in ((poles,pole_model,pole_threshold,"pole"),(lines,line_model,line_threshold,"line")):
        if df.empty:
            df["refiner_probability"]=[]; df["physical_ok"]=[]; df["component_accept"]=[]; continue
        x=df.reindex(columns=FEATURE_COLUMNS,fill_value=0.).replace([np.inf,-np.inf],np.nan).fillna(0.).to_numpy(float)
        df["refiner_probability"]=model.predict_proba(x)[:,1]
        if name=="pole":
            df["physical_ok"]=[pole_physical_ok(r,**physical_cfg.get("pole",{})) for _,r in df.iterrows()]
        else:
            df["physical_ok"]=[line_physical_ok(r,**physical_cfg.get("line",{})) for _,r in df.iterrows()]
        df["component_accept"]=(df.refiner_probability>=thr)&df.physical_ok
    return poles,lines


def load_refiner_bundle(path):
    return joblib.load(path)
