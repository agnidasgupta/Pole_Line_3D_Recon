#!/usr/bin/env python3
"""V6.2 slice-local Stage-2 geometry/refiner utilities. No world metadata."""
from __future__ import annotations
import numpy as np, pandas as pd

LOCAL_FEATURE_COLUMNS=[
    "n_voxels","score_mean","score_std","score_p10","score_p50","score_p90","score_max",
    "vertical_head_mean","horizontal_head_mean","x_span_ft","y_span_ft","z_span_ft","horizontal_span_ft",
    "bbox_density","center_z_ft","min_z_ft","max_z_ft","principal_dx","principal_dy","principal_dz",
    "principal_verticality","linearity","planarity","scattering","radius_p50_ft","radius_p90_ft",
    "xy_path_length_ft","xy_endpoint_distance_ft","xy_tortuosity","quadratic_rmse_ft","sag_estimate_ft",
    "touches_xy_edge","edge_distance_vox","partial_pole_candidate","exact_gt_fraction","near_gt_fraction"
]


def pole_local_physical_ok(row,min_height_ft=10.,max_height_ft=90.,max_radius_ft=3.5,min_verticality=.65,max_tilt_ratio=.65,
                           partial_min_height_ft=4.):
    h=float(row.z_span_ft); r=float(row.radius_p90_ft); hs=float(row.horizontal_span_ft); edge=bool(row.get("touches_xy_edge",False))
    minh=partial_min_height_ft if edge else min_height_ft
    return bool(minh<=h<=max_height_ft and r<=max_radius_ft and float(row.principal_verticality)>=min_verticality and hs/max(h,1e-6)<=max_tilt_ratio)


def line_local_physical_ok(row,min_horizontal_span_ft=.5,max_verticality=.85,max_vertical_horizontal=2.5,max_tortuosity=4.0):
    """Loose *hard plausibility* gate for Stage 2.

    V6.2 previously used this as a strict line-definition rule, which discarded
    short/occluded but legitimate wire fragments before Stage 3 could use session
    geometry.  The refiner now carries most of the discrimination; this gate only
    rejects clearly impossible predominantly-vertical or extremely tortuous objects.
    """
    h=float(row.horizontal_span_ft); z=float(row.z_span_ft)
    return bool(h>=min_horizontal_span_ft and float(row.principal_verticality)<=max_verticality and
                z/max(h,1e-6)<=max_vertical_horizontal and float(row.xy_tortuosity)<=max_tortuosity)


def assign_noise_tolerant_targets(df,class_name):
    """Use GT only to confirm positives. Plausible GT disagreements are ambiguous, not negatives."""
    d=df.copy()
    if d.empty:
        d["target"]=pd.Series(dtype=int); d["target_weight"]=pd.Series(dtype=float); d["target_source"]=pd.Series(dtype=str)
        return d
    if class_name=="pole":
        physical=np.array([pole_local_physical_ok(r) for _,r in d.iterrows()])
        pos=(d.near_gt_fraction.to_numpy(float)>=.30)|(d.exact_gt_fraction.to_numpy(float)>=.20)
    else:
        physical=np.array([line_local_physical_ok(r) for _,r in d.iterrows()])
        # Labels were generated within 2.5 voxels of an idealized line; near-GT overlap is more reliable than exact overlap.
        pos=(d.near_gt_fraction.to_numpy(float)>=.20)|(d.exact_gt_fraction.to_numpy(float)>=.10)
    target=np.full(len(d),-1,np.int8); source=np.full(len(d),"ambiguous_gt_disagreement",object); weight=np.zeros(len(d),np.float32)
    target[pos]=1; source[pos]="gt_confirmed_positive"; weight[pos]=1.0
    # Only clear geometric contradictions are used as negatives. A plausible V4/V6 prediction on GT background is ignored.
    neg=(~pos)&(~physical)
    target[neg]=0; source[neg]="geometry_confirmed_negative"; weight[neg]=.75
    d["local_physical_ok"]=physical; d["target"]=target; d["target_weight"]=weight; d["target_source"]=source
    return d


def add_edge_features(df,points_by_id,grid_size,edge_width_vox=10):
    d=df.copy(); gx,gy,_=map(int,grid_size); edge=[]; dist=[]
    for _,r in d.iterrows():
        pts=points_by_id.get(str(r.component_id),np.empty((0,3),int))
        if len(pts):
            ed=float(np.min(np.column_stack([pts[:,0],gx-1-pts[:,0],pts[:,1],gy-1-pts[:,1]])))
        else: ed=999.
        dist.append(ed); edge.append(ed<=edge_width_vox)
    d["edge_distance_vox"]=dist; d["touches_xy_edge"]=edge
    d["partial_pole_candidate"]=(d.class_name.eq("pole")&d.touches_xy_edge&(d.principal_verticality>=.65)).astype(int)
    return d
