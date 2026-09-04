#!/usr/bin/env python3
"""Learn electrical-safe Stage-1 track geometry from Velasco without GT.

0-19: stable conductor thickness/separation.
20-39: same-lane fragmentation gap scale after excluding overlap/cross-lane pairs.
The resulting profile is frozen for target and all-session Stage 2 runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4_stage_contracts import STAGE1_MANIFEST_COLUMNS, load_stage1_artifact, stage1_paths
from v4_realtime_core import load_calibration
from v4_stage2_stage1_electrical_tracks import (
    candidate_fragment_bridges,
    connected_components_26,
    describe_fragment,
    resolve_deployed_stage1_labels,
    _angle_deg,
)

HOST_OUTPUT_PREFIX = Path('/workspace/voxel_poleline/outputs')
CONTAINER_OUTPUT_PREFIX = Path('/outputs')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--stage1_dir', required=True)
    p.add_argument('--calibration_json', required=True)
    p.add_argument('--session_filter', default='VELASCO_CUT_CP/session1')
    p.add_argument('--reference_min', type=int, default=0)
    p.add_argument('--reference_max', type=int, default=19)
    p.add_argument('--fragment_min', type=int, default=20)
    p.add_argument('--fragment_max', type=int, default=39)
    p.add_argument('--voxel_size_ft', type=float, default=.5)
    p.add_argument('--grid_size', type=int, nargs=3, default=[400,400,200])
    p.add_argument('--max_gap_cap_ft', type=float, default=15.0)
    p.add_argument('--output_dir', required=True)
    return p.parse_args()


def map_host(p: Path) -> Path:
    s = str(p); prefix = str(HOST_OUTPUT_PREFIX)
    if s.startswith(prefix + '/'):
        return Path(str(CONTAINER_OUTPUT_PREFIX) + s[len(prefix):])
    return p


def resolve_artifacts(row: Any, root: Path):
    npz = map_host(Path(str(row.stage1_npz)))
    meta = map_host(Path(str(row.stage1_meta_json)))
    if npz.is_file() and meta.is_file():
        return npz, meta
    return stage1_paths(root, str(row.relative_path))


def pctl(values, q, default):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.quantile(a, q)) if len(a) else float(default)


def main():
    a = parse_args()
    if a.reference_min > a.reference_max or a.fragment_min > a.fragment_max:
        raise ValueError('invalid ordinal windows')
    root = Path(a.stage1_dir).resolve()
    out = Path(a.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    cal = load_calibration(a.calibration_json)
    mf = pd.read_csv(root / 'stage1_manifest.csv')
    missing = sorted(set(STAGE1_MANIFEST_COLUMNS)-set(mf.columns))
    if missing:
        raise RuntimeError(f'manifest missing {missing}')
    mf = mf[mf['group_id'].astype(str).eq(a.session_filter) & mf['status'].astype(str).eq('completed')].copy()
    mf['slice_seq'] = pd.to_numeric(mf['slice_seq'], errors='raise').astype(int)
    mf = mf.sort_values('slice_seq', kind='stable').reset_index(drop=True)
    mf['session_ordinal'] = np.arange(len(mf), dtype=int)
    need = max(a.reference_max, a.fragment_max)
    if len(mf) <= need:
        raise RuntimeError(f'{a.session_filter} has {len(mf)} slices; ordinal {need} unavailable')

    rows=[]; ref_radii=[]; ref_spacings=[]; target_fragment_sets=[]
    for r in mf.itertuples(index=False):
        ordn = int(r.session_ordinal)
        if not (a.reference_min <= ordn <= a.reference_max or a.fragment_min <= ordn <= a.fragment_max):
            continue
        npz, meta = resolve_artifacts(r, root)
        item, pred, _ = load_stage1_artifact(npz, meta)
        coords = np.asarray(item['coords'], dtype=np.int32)
        labels, source = resolve_deployed_stage1_labels(pred, cal)
        idx, comps = connected_components_26(coords, labels == 2, tuple(a.grid_size))
        frags = [describe_fragment(i,c,coords,a.voxel_size_ft) for i,c in enumerate(comps)]
        longf = [f for f in frags if len(f.voxel_indices)>=8 and f.horizontal_span_ft>=3.0]
        isolated = sum(1 for c in comps if len(c)==1)
        rows.append({
            'session_ordinal':ordn,'slice_seq':int(r.slice_seq),
            'stage1_line_voxels':int(len(idx)),'raw_components':int(len(comps)),
            'singleton_components':int(isolated),'long_components':int(len(longf)),
            'label_source':source,
        })
        if a.reference_min <= ordn <= a.reference_max:
            ref_radii.extend(float(f.radius_p90_ft) for f in longf)
            for i in range(len(longf)):
                for j in range(i+1,len(longf)):
                    fa, fb = longf[i], longf[j]
                    if _angle_deg(fa.axis_xy, fb.axis_xy) > 8.0:
                        continue
                    axis = fa.axis_xy + (fb.axis_xy if np.dot(fa.axis_xy,fb.axis_xy)>=0 else -fb.axis_xy)
                    axis = axis / max(float(np.linalg.norm(axis)),1e-12)
                    normal=np.array([-axis[1],axis[0]])
                    spacing=abs(float(np.dot((fb.center_xy-fa.center_xy)*a.voxel_size_ft,normal)))
                    if .75 <= spacing <= 30.0:
                        ref_spacings.append(spacing)
        else:
            target_fragment_sets.append((ordn,int(r.slice_seq),coords,frags))

    radius95 = pctl(ref_radii,.95,.5)
    spacing10 = pctl(ref_spacings,.10,4.0)

    # Lane identity must be much tighter than inter-conductor spacing.
    max_lane = max(0.5, min(0.75, 1.5*radius95, 0.20*spacing10))
    max_track_radius = max(0.75, min(1.25, 2.0*radius95, 0.40*spacing10))
    provisional={
        'max_gap_ft':float(a.max_gap_cap_ft),
        'max_lane_offset_ft':float(max_lane),
        'max_track_radius_ft':float(max_track_radius),
        'max_longitudinal_overlap_ft':float(a.voxel_size_ft*2.0),
        'max_axis_angle_deg':10.0,
        'max_bridge_angle_deg':15.0,
        'max_vertical_gap_ft':6.0,
        'pole_bridge_guard_radius_ft':6.0,
        'pole_attach_radius_ft':8.0,
        'pole_attach_max_angle_deg':45.0,
        'pole_attach_min_height_fraction':0.35,
        'pole_surface_standoff_min_ft':0.5,
        'min_fragment_voxels':2,
        'vertex_bin_ft':1.0,
    }

    compatible=[]
    for ordn,seq,coords,frags in target_fragment_sets:
        for c in candidate_fragment_bridges(frags,coords,a.voxel_size_ft,provisional,poles=None):
            if c['passed']:
                compatible.append({**c,'session_ordinal':ordn,'slice_seq':seq})

    gaps=[r['gap_ft'] for r in compatible]
    vertical=[r['vertical_gap_ft'] for r in compatible]
    learned_gap=max(4.0,min(float(a.max_gap_cap_ft),pctl(gaps,.95,8.0)))
    learned_vertical=max(3.0,min(6.0,pctl(vertical,.95,4.0)))

    profile={
        'profile_version':'velasco-stage1-electrical-v10-20260904',
        'learned_from_session':a.session_filter,
        'reference_ordinals':[a.reference_min,a.reference_max],
        'fragmentation_ordinals':[a.fragment_min,a.fragment_max],
        'runtime_gt_usage':False,
        'pole_pair_inference':False,
        'max_gap_ft':round(learned_gap,6),
        'max_lane_offset_ft':round(max_lane,6),
        'max_track_radius_ft':round(max_track_radius,6),
        'max_longitudinal_overlap_ft':round(a.voxel_size_ft*2.0,6),
        'max_axis_angle_deg':10.0,
        'max_bridge_angle_deg':15.0,
        'max_vertical_gap_ft':round(learned_vertical,6),
        'pole_bridge_guard_radius_ft':6.0,
        'pole_attach_radius_ft':8.0,
        'pole_attach_max_angle_deg':45.0,
        'pole_attach_min_height_fraction':0.35,
        'pole_surface_standoff_min_ft':0.5,
        'min_fragment_voxels':2,
        'vertex_bin_ft':1.0,
        'reference_track_radius_p95_ft':round(radius95,6),
        'reference_parallel_spacing_p10_ft':round(spacing10,6),
        'compatible_target_gap_p95_ft':round(pctl(gaps,.95,0.0),6),
        'compatible_target_pairs':int(len(compatible)),
        'electrical_rule_parallel_lanes_never_merge':True,
        'electrical_rule_near_pole_line_to_line_bridge':False,
    }
    (out/'selected_electrical_profile.json').write_text(json.dumps(profile,indent=2,sort_keys=True)+'\n')
    frame=pd.DataFrame(rows).sort_values('session_ordinal')
    frame.to_csv(out/'velasco_window_fragmentation.csv',index=False)
    pd.DataFrame(compatible).to_csv(out/'velasco_same_lane_fragment_pairs.csv',index=False)
    ref=frame[frame.session_ordinal.between(a.reference_min,a.reference_max)]
    tar=frame[frame.session_ordinal.between(a.fragment_min,a.fragment_max)]
    def stats(x):
        vox=max(int(x.stage1_line_voxels.sum()),1)
        return {
            'slices':int(len(x)),
            'line_voxels':int(x.stage1_line_voxels.sum()),
            'raw_components':int(x.raw_components.sum()),
            'components_per_1000_line_voxels':float(1000*x.raw_components.sum()/vox),
            'singleton_components':int(x.singleton_components.sum()),
        }
    report={
        'reference_window':stats(ref),
        'fragmentation_window':stats(tar),
        'learned_profile':profile,
        'interpretation':(
            'V9 correctly bridged fragmented Stage1 conductors but could merge nearby parallel or converging lanes. '
            'V10 requires end-to-end same-lane continuation, rejects longitudinally overlapping lanes, constrains '
            'track lateral drift, forbids line-to-line bridges near detected poles, and allows separate track endpoints '
            'to attach to detected pole surfaces.'
        ),
    }
    (out/'velasco_electrical_profile_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    with open(out/'velasco_electrical_profile_report.txt','w') as f:
        f.write('V10 STAGE1 ELECTRICAL TRACK LEARNING\n')
        f.write(json.dumps(report,indent=2,sort_keys=True)); f.write('\n')
    print('V10_STAGE1_ELECTRICAL_PROFILE_OK')
    print(json.dumps(profile,sort_keys=True))

if __name__=='__main__': main()
