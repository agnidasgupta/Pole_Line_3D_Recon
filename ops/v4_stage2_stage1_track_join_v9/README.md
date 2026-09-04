# V4 Stage2 Stage1 Track Join V9

Purpose: turn deployed Stage-1 line-labelled voxels into coherent Stage-2 conductor tracks without creating lines from pole pairs.

## Root cause addressed

V8.1 preserved every deployed Stage-1 class-2 voxel, but emitted one Stage-2 line component per exact 26-neighbour voxel pair and explicitly refused gap bridging. Dense Stage-1 occupancy therefore appeared joined, while the same physical conductor split by short missing label runs remained fragmented.

V9 keeps the same Stage-1 voxel set and production pole outputs, but changes line geometry generation:

1. Resolve deployed Stage-1 labels and select exactly class 2 voxels.
2. Build exact 26-neighbour components.
3. Learn stable line thickness/separation from Velasco session1 ordinals 0-19.
4. Measure the fragmentation-gap scale in ordinals 20-39.
5. Join only geometrically compatible class-2 fragments: short gap, small lateral offset, compatible axis, compatible bridge direction, plausible vertical gap.
6. Emit one polyline per joined conductor track.
7. Every bridge endpoint is an observed Stage-1 class-2 voxel. No synthetic voxel labels are created.
8. Poles are not enumerated or used to create line candidates.

The learned profile is fixed and reused for all other sessions.

## Canonical output contract

`/workspace/voxel_poleline/outputs/poleline_voxel_run_session_groups/v4_stage23_quality/full_run_v2_<UTC>/`

- `stage2/<SID>/`
- `stage3/` reserved empty
- `selection/`
- `logs/stage2/`
- `logs/stage3/` reserved empty
- `timing/stage2/`
- `status/`
- `RUN_INFO.txt`
- `session_map.tsv`

## Runtime invariants

- Stage1 class-2 voxel preservation = 1.0
- runtime GT use = false
- synthetic line voxels = 0
- pole-pair inference = false
- production poles preserved
- bridge endpoints are observed Stage1 class-2 voxels

## Main files

- `v4_stage2_stage1_track_join.py`
- `learn_velasco_stage1_track_profile.py`
- `run_v4_stage2_stage1_track_join.py`
- `self_test_stage1_track_join.py`
- `run_v4_stage2_stage1_track_join_v9_on_nebius.sh`
- `monitor_v4_stage2_stage1_track_join_v9_on_nebius.sh`
- `download_v4_stage2_stage1_track_join_v9_to_mac.sh`
