#!/usr/bin/env python3
"""Sparse, slice-local V4 Stage-2 component extraction.

Only current-slice local voxel coordinates and V4 scores are used at runtime.
GT coordinates are accepted only by offline mining to construct training targets.
No center/world/session geometry is used here.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from v4_stage2_local import add_edge_features, assign_noise_tolerant_targets


def _finite_points(points):
    p=np.asarray(points,dtype=np.float64)
    return p[np.isfinite(p).all(axis=1)]


def pca_geometry(points_xyz: np.ndarray, voxel_size=.5):
    p=_finite_points(points_xyz)
    if len(p)<2:
        q=p[0] if len(p) else np.zeros(3)
        return dict(principal=np.array([0.,0.,1.]),linearity=0.,planarity=0.,scattering=0.,
                    radius_p50_ft=0.,radius_p90_ft=0.,endpoints=np.vstack([q,q]),
                    xy_path_length_ft=0.,xy_endpoint_distance_ft=0.,xy_tortuosity=1.,
                    quadratic_rmse_ft=0.,sag_estimate_ft=0.)
    c=np.median(p,axis=0); centered=p-c
    try:
        _,_,vh=np.linalg.svd(centered,full_matrices=False); direction=vh[0]; direction/=max(np.linalg.norm(direction),1e-9)
    except np.linalg.LinAlgError: direction=np.array([0.,0.,1.])
    cov=np.cov(centered.T) if len(p)>2 else np.zeros((3,3))
    try: ev=np.sort(np.linalg.eigvalsh(cov))[::-1]
    except np.linalg.LinAlgError: ev=np.zeros(3)
    e1,e2,e3=[float(max(x,0.)) for x in ev]
    linearity=(e1-e2)/max(e1,1e-9); planarity=(e2-e3)/max(e1,1e-9); scattering=e3/max(e1,1e-9)
    t=centered@direction; fitted=c+np.outer(t,direction); radial=np.linalg.norm(p-fitted,axis=1)*voxel_size
    xy=p[:,:2]; cxy=np.median(xy,axis=0)
    try: _,_,vhxy=np.linalg.svd(xy-cxy,full_matrices=False); dxy=vhxy[0]
    except np.linalg.LinAlgError: dxy=np.array([1.,0.])
    dxy/=max(np.linalg.norm(dxy),1e-9); s=(xy-cxy)@dxy; order=np.argsort(s); op=p[order]
    endpoints=np.vstack([op[0],op[-1]])
    dif=np.diff(op[:,:2],axis=0); path=float(np.linalg.norm(dif,axis=1).sum()*voxel_size) if len(op)>1 else 0.
    endpoint_d=float(np.linalg.norm(op[-1,:2]-op[0,:2])*voxel_size) if len(op)>1 else 0.; tort=path/max(endpoint_d,1e-6)
    rmse=sag=0.
    if len(p)>=6:
        sf=np.asarray(s,float)*voxel_size; zf=np.asarray(p[:,2],float)*voxel_size; span=float(np.ptp(sf))
        if np.isfinite(sf).all() and np.isfinite(zf).all() and span>=max(voxel_size*2,1e-6) and np.unique(np.round(sf,6)).size>=3:
            try:
                mid=.5*(sf.min()+sf.max()); half=max(.5*span,1e-9); u=(sf-mid)/half
                A=np.column_stack([u*u,u,np.ones_like(u)])
                if np.isfinite(np.linalg.cond(A)) and np.linalg.cond(A)<1e8:
                    coef,*_=np.linalg.lstsq(A,zf,rcond=None); fit=A@coef; rmse=float(np.sqrt(np.mean((fit-zf)**2)))
                    sag=float(max(0.,.5*((coef[0]-coef[1]+coef[2])+(coef[0]+coef[1]+coef[2]))-coef[2]))
            except (np.linalg.LinAlgError,ValueError,FloatingPointError): pass
    return dict(principal=direction,linearity=linearity,planarity=planarity,scattering=scattering,
                radius_p50_ft=float(np.quantile(radial,.5)),radius_p90_ft=float(np.quantile(radial,.9)),endpoints=endpoints,
                xy_path_length_ft=path,xy_endpoint_distance_ft=endpoint_d,xy_tortuosity=tort,
                quadratic_rmse_ft=rmse,sag_estimate_ft=sag)


# 13 positive half-neighborhood offsets => all 26 neighbors without duplicate edges.
_HALF26=[]
for dz in (-1,0,1):
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if (dx,dy,dz)==(0,0,0): continue
            if (dz,dy,dx)>(0,0,0): _HALF26.append((dx,dy,dz))


def sparse_connected_labels(coords: np.ndarray, grid_size=(400,400,200)) -> Tuple[np.ndarray,int]:
    """Exact 26-connectivity labels for unique sparse XYZ coordinates.

    Returns labels 1..K aligned with input order. No full dense 3-D volume is allocated.
    """
    c=np.asarray(coords,np.int32); n=len(c)
    if n==0: return np.zeros(0,np.int32),0
    gx,gy,gz=map(int,grid_size)
    key=((c[:,2].astype(np.int64)*gy+c[:,1])*gx+c[:,0])
    order=np.argsort(key); skey=key[order]
    srcs=[]; dsts=[]
    for dx,dy,dz in _HALF26:
        q=c+np.array([dx,dy,dz],np.int32)
        valid=(q[:,0]>=0)&(q[:,0]<gx)&(q[:,1]>=0)&(q[:,1]<gy)&(q[:,2]>=0)&(q[:,2]<gz)
        ii=np.flatnonzero(valid)
        if not len(ii): continue
        qv=q[ii]; qkey=((qv[:,2].astype(np.int64)*gy+qv[:,1])*gx+qv[:,0])
        pos=np.searchsorted(skey,qkey); ok=pos<n
        hit=np.zeros(len(pos),bool); hit[ok]=skey[pos[ok]]==qkey[ok]
        if np.any(hit):
            jj=order[pos[hit]]; aa=ii[hit]
            srcs.append(aa); dsts.append(jj)
    if not srcs:
        return np.arange(1,n+1,dtype=np.int32),n
    a=np.concatenate(srcs); b=np.concatenate(dsts)
    rows=np.concatenate([a,b,np.arange(n)]); cols=np.concatenate([b,a,np.arange(n)])
    graph=coo_matrix((np.ones(len(rows),np.uint8),(rows,cols)),shape=(n,n)).tocsr()
    k,lab=connected_components(graph,directed=False,return_labels=True)
    return lab.astype(np.int32)+1,int(k)


def _component_frame(coords, score, class_name, voxel_size=.5, gt_points=None, min_voxels=3):
    labels,n=sparse_connected_labels(coords)
    rows=[]; points={}
    if n==0: return pd.DataFrame(),labels,points
    gt_tree=cKDTree(np.asarray(gt_points,float)) if gt_points is not None and len(gt_points) else None
    for cid in range(1,n+1):
        idx=np.flatnonzero(labels==cid)
        if len(idx)<int(min_voxels): continue
        pts=coords[idx].astype(np.int32); vals=np.asarray(score,dtype=np.float32)[idx]
        geo=pca_geometry(pts,voxel_size); mins=pts.min(0); maxs=pts.max(0); spans=(maxs-mins+1)*voxel_size
        bbox=float(np.prod(maxs-mins+1)); d=np.asarray(geo['principal'],float)
        exact=near=0.
        if gt_tree is not None:
            dist,_=gt_tree.query(pts.astype(float),k=1,workers=-1); exact=float(np.mean(dist<=1e-9)); near=float(np.mean(dist<=2.5))
        pid=f"{class_name[0].upper()}{cid:05d}"
        # V4 has no explicit orientation heads. Use the component's local PCA orientation
        # as the corresponding slice-local orientation features.
        vert=float(abs(d[2])); horiz=float(math.sqrt(max(0.,1.-vert*vert)))
        ep=geo['endpoints']
        row=dict(component_id=pid,class_name=class_name,n_voxels=int(len(pts)),
                 score_mean=float(vals.mean()),score_std=float(vals.std()),score_p10=float(np.quantile(vals,.1)),
                 score_p50=float(np.quantile(vals,.5)),score_p90=float(np.quantile(vals,.9)),score_max=float(vals.max()),
                 vertical_head_mean=vert,horizontal_head_mean=horiz,
                 x_span_ft=float(spans[0]),y_span_ft=float(spans[1]),z_span_ft=float(spans[2]),horizontal_span_ft=float(np.linalg.norm(spans[:2])),
                 bbox_density=float(len(pts)/max(bbox,1.)),center_x=float(pts[:,0].mean()),center_y=float(pts[:,1].mean()),center_z=float(pts[:,2].mean()),
                 center_z_ft=float(pts[:,2].mean()*voxel_size),min_z_ft=float(mins[2]*voxel_size),max_z_ft=float(maxs[2]*voxel_size),
                 principal_dx=float(d[0]),principal_dy=float(d[1]),principal_dz=float(d[2]),principal_verticality=vert,
                 linearity=float(geo['linearity']),planarity=float(geo['planarity']),scattering=float(geo['scattering']),
                 radius_p50_ft=float(geo['radius_p50_ft']),radius_p90_ft=float(geo['radius_p90_ft']),
                 xy_path_length_ft=float(geo['xy_path_length_ft']),xy_endpoint_distance_ft=float(geo['xy_endpoint_distance_ft']),xy_tortuosity=float(geo['xy_tortuosity']),
                 quadratic_rmse_ft=float(geo['quadratic_rmse_ft']),sag_estimate_ft=float(geo['sag_estimate_ft']),
                 endpoint1_x=float(ep[0,0]),endpoint1_y=float(ep[0,1]),endpoint1_z=float(ep[0,2]),
                 endpoint2_x=float(ep[1,0]),endpoint2_y=float(ep[1,1]),endpoint2_z=float(ep[1,2]),
                 exact_gt_fraction=exact,near_gt_fraction=near)
        rows.append(row); points[pid]=pts
    return pd.DataFrame(rows),labels,points


def extract_sparse_components(item: Dict, scores: Dict[str,np.ndarray], grid_size=(400,400,200), voxel_size=.5,
                              pole_threshold=.15,line_threshold=.08,line_weak_threshold=.04,line_competition_ratio=.55,
                              pole_min_voxels=4,line_min_voxels=3,gt_points=None,edge_width_vox=10):
    coords=np.asarray(item['coords'],np.int32); ps=np.asarray(scores['pole'],np.float32); ls=np.asarray(scores['line'],np.float32)
    if not (len(coords)==len(ps)==len(ls)): raise ValueError('coords and row scores must align')
    pm=(ps>=float(pole_threshold))&(ps>=ls*.8)
    weak=(ls>=float(line_weak_threshold))&(ls>=ps*float(line_competition_ratio))
    strong=(ls>=float(line_threshold))&(ls>=ps*float(line_competition_ratio))

    pcoords=coords[pm]; pscore=ps[pm]
    poles,plocal,ppoints=_component_frame(pcoords,pscore,'pole',voxel_size,(gt_points or {}).get(1),pole_min_voxels)

    wcoords=coords[weak]; wscore=ls[weak]
    wlabels,nw=sparse_connected_labels(wcoords,grid_size)
    if nw and np.any(strong):
        # Map strong original rows into the weak-array order by sparse coordinate key.
        gx,gy,gz=map(int,grid_size)
        wk=((wcoords[:,2].astype(np.int64)*gy+wcoords[:,1])*gx+wcoords[:,0]); sk=np.sort(wk); ordw=np.argsort(wk)
        sc=coords[strong]; skey=((sc[:,2].astype(np.int64)*gy+sc[:,1])*gx+sc[:,0]); pos=np.searchsorted(sk,skey); ok=(pos<len(sk))
        hit=np.zeros(len(pos),bool); hit[ok]=sk[pos[ok]]==skey[ok]
        strong_w=ordw[pos[hit]] if np.any(hit) else np.zeros(0,np.int64)
        keep_ids=np.unique(wlabels[strong_w]) if len(strong_w) else np.zeros(0,np.int32)
        keep=np.isin(wlabels,keep_ids)
    else: keep=np.zeros(len(wcoords),bool)
    lcoords=wcoords[keep]; lscore=wscore[keep]
    lines,llocal,lpoints=_component_frame(lcoords,lscore,'line',voxel_size,(gt_points or {}).get(2),line_min_voxels)

    poles=add_edge_features(poles,ppoints,grid_size,edge_width_vox) if not poles.empty else poles
    lines=add_edge_features(lines,lpoints,grid_size,edge_width_vox) if not lines.empty else lines
    if gt_points is not None:
        poles=assign_noise_tolerant_targets(poles,'pole'); lines=assign_noise_tolerant_targets(lines,'line')
    return {'poles':poles,'lines':lines,'pole_points':ppoints,'line_points':lpoints,
            'candidate_counts':{'pole_voxels':int(pm.sum()),'line_weak_voxels':int(weak.sum()),'line_kept_voxels':int(keep.sum())}}
