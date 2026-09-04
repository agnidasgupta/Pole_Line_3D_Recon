#!/usr/bin/env python3
import numpy as np
from v4_stage2_stage1_track_join import build_track_join_outputs


def main():
    pts=[]
    # Same conductor, two observed fragments separated by a 2 ft missing run.
    pts += [(x,10,20) for x in range(0,6)]
    pts += [(x,10,20) for x in range(9,16)]
    # Parallel conductor 3 ft away: must remain separate.
    pts += [(x,16,20) for x in range(0,16)]
    # Off-axis fragment: must remain separate.
    pts += [(25,y,20) for y in range(0,6)]
    coords=np.asarray(pts,dtype=np.int32)
    labels=np.full(len(coords),2,dtype=np.int8)
    scores=np.full(len(coords),.9,dtype=np.float32)
    profile={
        'max_gap_ft':4.0,
        'max_lateral_ft':1.0,
        'max_axis_angle_deg':12.0,
        'max_bridge_angle_deg':20.0,
        'max_vertical_gap_ft':3.0,
        'min_fragment_voxels':2,
        'vertex_bin_ft':1.0,
    }
    out=build_track_join_outputs(coords,scores,labels,'synthetic',1,(400,400,200),.5,profile)
    a=out['audit']
    assert a['stage1_inferred_line_voxels']==len(coords)
    assert a['accepted_stage1_line_voxels']==len(coords)
    assert a['stage1_to_stage2_voxel_preservation']==1.0
    assert a['synthetic_line_voxels']==0
    assert a['pole_pair_inference'] is False
    assert a['selected_fragment_bridges']==1, a
    assert a['joined_stage2_tracks']==3, a
    selected=out['selected_bridges']
    assert len(selected)==1
    assert selected[0]['gap_ft'] <= 4.0
    # No selected bridge may jump to the parallel y=16 conductor.
    ia=selected[0]['a_endpoint_index']; ib=selected[0]['b_endpoint_index']
    assert coords[ia,1]==10 and coords[ib,1]==10
    print('V4_STAGE2_STAGE1_TRACK_JOIN_SELF_TEST_OK')

if __name__=='__main__': main()
