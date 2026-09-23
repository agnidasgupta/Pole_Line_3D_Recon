#!/usr/bin/env python3
from __future__ import annotations
import inspect,re
from pathlib import Path
import numpy as np
from scipy import ndimage
import torch.nn as nn

from voxel_common import MultiHeadVoxelNet3D
from v4_realtime_core import active_core_groups,build_v4_patch,build_v4_data_patch,assemble_v4_channels_gpu
from v4_sparse_components import sparse_connected_labels
from v4_stage2_local import LOCAL_FEATURE_COLUMNS


def partition_signature(labels):
    d={}
    for i,x in enumerate(labels): d.setdefault(int(x),[]).append(i)
    return sorted(sorted(v) for k,v in d.items() if k>0)


def connectivity_test():
    rng=np.random.default_rng(42); grid=(24,20,12)
    xx,yy,zz=np.meshgrid(np.arange(grid[0]),np.arange(grid[1]),np.arange(grid[2]),indexing='ij')
    allc=np.column_stack([xx.ravel(),yy.ravel(),zz.ravel()])
    take=rng.choice(len(allc),size=700,replace=False); c=allc[take].astype(np.int32)
    sl,k=sparse_connected_labels(c,grid)
    dense=np.zeros((grid[2],grid[1],grid[0]),np.uint8); dense[c[:,2],c[:,1],c[:,0]]=1
    dl,n=ndimage.label(dense,structure=np.ones((3,3,3),np.uint8)); old=dl[c[:,2],c[:,1],c[:,0]]
    assert partition_signature(sl)==partition_signature(old), 'sparse 26-connectivity differs from ndimage.label'
    return k


def core_test():
    rng=np.random.default_rng(3); c=np.column_stack([rng.integers(0,400,5000),rng.integers(0,400,5000),rng.integers(0,200,5000)]).astype(np.int32)
    c=np.unique(c,axis=0); groups=active_core_groups(c,(400,400,200),48); rows=np.concatenate([g['rows'] for g in groups])
    assert len(rows)==len(c) and np.array_equal(np.sort(rows),np.arange(len(c))), 'active-core schedule lost/duplicated rows'
    return len(groups)


def channel_assembly_test():
    import torch
    rng=np.random.default_rng(7); coords=np.unique(np.column_stack([rng.integers(0,400,300),rng.integers(0,400,300),rng.integers(0,200,300)]),axis=0).astype(np.int32); order=np.argsort(coords[:,2]); coords=coords[order]
    item={'coords':coords,'dist_values':rng.random(len(coords),dtype=np.float32),'z_sorted':coords[:,2]}
    center=np.array([123,250,91]); old=build_v4_patch(item,center,(400,400,200),64,True,True)
    data=build_v4_data_patch(item,center,64,True); new=assemble_v4_channels_gpu(torch.from_numpy(data[None]),[center],(400,400,200),64,True,True)[0].numpy()
    err=float(np.max(np.abs(old-new))); assert err<=2e-7, err; return err

def architecture_test():
    m=MultiHeadVoxelNet3D(in_ch=5,base=16)
    gn=sum(isinstance(x,nn.GroupNorm) for x in m.modules()); bn=sum(isinstance(x,(nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d,nn.SyncBatchNorm)) for x in m.modules())
    assert gn>0 and bn==0, (gn,bn)
    forbidden=[x for x in LOCAL_FEATURE_COLUMNS if x.startswith('world_') or x in {'center_x','center_y','center_z'} or x in {'exact_gt_fraction','near_gt_fraction'}]
    assert not forbidden, forbidden
    return gn,bn


def stage3_contract_test():
    s=Path(__file__).with_name('reconstruct_v4_stage3.py').read_text()
    required=['default=450.0','default=9','default=10.0','default=50.0','default=8.0','default=12.0','default=20.0','--latest_slice requires --session_filter','a.latest_slice - a.max_span_slices']
    miss=[x for x in required if x not in s]; assert not miss, f'Stage3 contract strings missing: {miss}'


def stage12_world_guard():
    # The model/refiner feature contract is the authoritative guard. Local component
    # centroids such as center_z_ft are allowed; session/world centers are not.
    forbidden={'center_x','center_y','center_z','world_x','world_y','world_z','exact_gt_fraction','near_gt_fraction'}
    assert not (forbidden & set(LOCAL_FEATURE_COLUMNS)), forbidden & set(LOCAL_FEATURE_COLUMNS)
    src=Path(__file__).with_name('v4_stage2_runtime.py').read_text()
    assert 'extract_center_metadata' not in src, 'Stage2 runtime must not access Stage3 center metadata'


def main():
    k=connectivity_test(); cores=core_test(); err=channel_assembly_test(); gn,bn=architecture_test(); stage3_contract_test(); stage12_world_guard()
    print(f'V4_REALTIME_SMOKE_OK components={k} active_cores={cores} coord_channel_max_error={err:.3g} groupnorm={gn} batchnorm={bn}')

if __name__=='__main__': main()
