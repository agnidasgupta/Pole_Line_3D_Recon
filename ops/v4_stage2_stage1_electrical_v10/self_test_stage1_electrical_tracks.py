#!/usr/bin/env python3
import numpy as np
import pandas as pd

from v4_stage2_stage1_electrical_tracks import build_electrical_track_outputs


def line(xs, ys, z):
    return np.asarray([[int(x), int(y), int(z)] for x, y in zip(xs, ys)], dtype=np.int32)


def main():
    # Same physical conductor with one short missing run.
    a = line(range(0, 11), [10]*11, 20)
    b = line(range(14, 25), [10]*11, 20)

    # Nearby almost-parallel conductor; overlaps longitudinally and must never merge.
    c = line(range(0, 25), [12]*25, 20)

    # Two distinct conductors that converge toward the same pole from different angles.
    xs = list(range(30, 39))
    d = line(xs, [26,26,27,27,28,28,29,29,30], 22)
    e = line(xs, [36,36,35,35,34,34,33,33,32], 24)

    coords = np.vstack([a,b,c,d,e])
    labels = np.full(len(coords), 2, dtype=np.int8)
    scores = np.full(len(coords), 0.95, dtype=np.float32)
    poles = pd.DataFrame([{
        'file_id':'f','component_id':'P1','slice_seq':1,
        'refiner_probability':1.0,'touches_xy_edge':False,'radius_p90_ft':0.75,'verticality':1.0,
        'base_x':40.0,'base_y':31.0,'base_z':0.0,
        'top_x':40.0,'top_y':31.0,'top_z':40.0,
        'height_ft':20.0,'tilt_ft':0.0,
    }])
    profile = {
        'max_gap_ft':8.0,
        'max_lane_offset_ft':0.75,
        'max_track_radius_ft':1.0,
        'max_longitudinal_overlap_ft':1.0,
        'max_axis_angle_deg':10.0,
        'max_bridge_angle_deg':15.0,
        'max_vertical_gap_ft':5.0,
        'pole_bridge_guard_radius_ft':6.0,
        'pole_attach_radius_ft':8.0,
        'pole_attach_max_angle_deg':45.0,
        'pole_attach_min_height_fraction':0.35,
        'pole_surface_standoff_min_ft':0.5,
        'min_fragment_voxels':2,
        'vertex_bin_ft':1.0,
    }
    out = build_electrical_track_outputs(
        coords, scores, labels, poles, 'f', 1, (100,100,100), 0.5, profile
    )
    assert out['audit']['stage1_inferred_line_voxels'] == len(coords)
    assert out['audit']['accepted_stage1_line_voxels'] == len(coords)
    assert out['audit']['stage1_to_stage2_voxel_preservation'] == 1.0
    assert out['audit']['synthetic_line_voxels'] == 0
    assert out['audit']['parallel_lane_merge_allowed'] is False
    assert out['audit']['line_to_line_bridge_near_pole_allowed'] is False

    # Exactly one bridge should connect a<->b; c must remain its own parallel lane.
    sel = out['selected_bridges']
    assert len(sel) == 1, sel
    assert sel[0]['lane_center_offset_ft'] <= profile['max_lane_offset_ft']
    assert sel[0]['longitudinal_overlap_ft'] <= profile['max_longitudinal_overlap_ft']

    # a+b, c, d, e => four electrically distinct Stage2 tracks.
    assert len(out['lines_rows']) == 4, out['track_rows']

    # The two converging lines should attach independently to the pole, not to one another.
    attachments = out['pole_attachment_rows']
    assert len(attachments) >= 2, attachments
    pole_rows = [r for r in attachments if r['pole_component_id'] == 'P1']
    assert len(pole_rows) >= 2, pole_rows
    anchors = {(round(r['anchor_x'],6),round(r['anchor_y'],6),round(r['anchor_z'],6)) for r in pole_rows}
    assert len(anchors) >= 2, anchors

    print('V4_STAGE2_STAGE1_ELECTRICAL_TRACK_SELF_TEST_OK')


if __name__ == '__main__':
    main()
