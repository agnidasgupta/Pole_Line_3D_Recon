# V4 Stage2 Stage1 Electrical Tracks V10

Accepted reference implementation of the strict, voxel-supported V10 Stage 2 electrical-line reconstruction.

This is a Stage2-only workflow. It consumes saved Stage1 sparse inference artifacts, does not rerun Stage1, does not run Stage3, and leaves production `v4/` unchanged. This directory contains no ONNX export path.

The behavior-preserving Opt1 performance experiment is documented separately in [`../v4_stage2_stage1_electrical_v10_opt/README.md`](../v4_stage2_stage1_electrical_v10_opt/README.md).

## Why V10

Earlier V9 fragment bridging could merge nearby conductors. It selected among fragment endpoints using permissive geometric tolerances and lacked a sufficiently strict pole-neighborhood topology rule. In particular, nearby parallel or converging conductors could be mistaken for one continuation.

The accepted strict V10 model therefore does not bridge disconnected Stage1 line components. Candidate-bridge diagnostics may still be emitted by the reference implementation, but no disconnected bridge may be selected into output geometry.

## Reconstruction method

Stage 2 resolves the deployed Stage1 sparse predictions and reconstructs electrical lines only from voxels assigned class `2`. The accepted production pole extraction remains unchanged.

For each slice, reconstruction:

1. Selects the Stage1 class-2 line voxels.
2. Partitions them into deterministic 26-neighbor connected components.
3. Converts each component into a voxel-adjacency graph.
4. Extracts ordered graph-diameter paths and splits them at sharp turns.
5. Partitions branch voxels so every inferred line voxel belongs to exactly one emitted track.
6. Simplifies path vertices only when sampled chords remain entirely inside the inferred Stage1 line-voxel support.
7. Attaches a line endpoint to a pole only when inferred line and pole voxels make direct contact.
8. Audits voxel identity, assignment completeness, topology and geometry support.

The reconstruction does not create synthetic line voxels. Separate conductors remain separate components, including when they independently contact the same pole. Their pole-attachment points remain distinct.

## Strict electrical and geometry rules

- Runtime line geometry starts only from deployed Stage1 `label == 2` voxels.
- Production pole detections are preserved.
- Disconnected Stage1 line components are never bridged.
- Every inferred line voxel is assigned to exactly one output track.
- Simplified geometry must have a Stage1 voxel-support fraction of `1.0`.
- Line-to-pole attachment requires direct inferred line/pole voxel contact.
- Separate conductor tracks may independently attach to the same pole surface.
- Pole attachments use distinct surface points derived from each track's contact and approach.
- No pole-pair enumeration is used.
- No runtime ground truth is used.
- No synthetic line voxels are introduced.
- Open line endpoints are preserved when no supported pole contact exists.

## Main entry points

- `v4_stage2_stage1_electrical_tracks.py` — reference reconstruction implementation.
- `run_v4_stage2_stage1_electrical_tracks.py` — per-session Stage 2 runner.
- `learn_velasco_stage1_electrical_profile.py` — fixed electrical profile learning from saved Stage1 outputs.
- `validate_v10_voxel_supported_stage2.py` — independent voxel-support validation.
- `run_v10_voxel_supported_fault_tolerant_stage2.sh` — fault-tolerant all-session driver.
- `launch_v10_voxel_supported_all_sessions.sh` — background launcher.
- `monitor_v10_voxel_supported_stage2.sh` — run monitor.
- `package_v10_voxel_supported_stage2_results.sh` — validated result packager.
- `download_v4_stage2_stage1_electrical_v10_to_mac.sh` — result download helper.

## Output contract

The all-session runner writes:

```text
/workspace/voxel_poleline/outputs/poleline_voxel_run_session_groups/
└── v4_stage23_quality/
    └── fault_tolerant_v10_<UTC>/
        ├── stage2/<SID>/
        ├── stage3/                  # reserved; Stage3 is not run
        ├── selection/
        ├── logs/stage2/
        ├── logs/stage3/             # reserved
        ├── timing/stage2/
        ├── status/
        ├── FILE_INVENTORY.txt
        ├── RUN_INFO.txt
        ├── session_map.tsv
        ├── PHASE2_STAGE2_OK.txt
        └── STAGE2_ONLY_COMPLETE.txt
```

`EXPECTED_SESSIONS` defaults to `30`. Successful completion requires every expected session to finish with no failed-session marker. Each accepted session receives an independent voxel-support validation report.

## Per-slice diagnostics

- `*_stage1_line_voxels.csv`
- `*_accepted_line_voxels.csv`
- `*_fragment_bridge_candidates.csv`
- `*_selected_fragment_bridges.csv`
- `*_stage1_electrical_tracks.csv`
- `*_pole_attachments.csv`
- `*_stage1_electrical_track_audit.json`

Important audit fields include:

- `stage1_inferred_line_voxels`
- `accepted_stage1_line_voxels`
- `stage1_to_stage2_voxel_preservation`
- `selected_fragment_bridges`
- `geometry_stage1_voxel_support_fraction`
- `geometry_outside_stage1_voxel_samples`
- `pole_attachments`
- `runtime_gt_usage`
- `synthetic_line_voxels`
- `disconnected_fragment_bridges_allowed`

## Validation and packaging

Before an all-session run, the driver compiles the Python files, runs `self_test_stage1_electrical_tracks.py`, learns the fixed Velasco profile and validates each completed session independently.

The packager requires:

- `STAGE2_ONLY_COMPLETE.txt`;
- 30 accepted-session markers;
- zero failed-session markers; and
- 30 voxel-support validation reports.

Only after those gates pass does it create `v10_voxel_supported_stage2_results_<UTC>.tar.gz`, its SHA-256 file and `/home/agni/LATEST_V10_VOXEL_SUPPORTED_STAGE2_ARCHIVE.txt`.
