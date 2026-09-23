# V4 Stage2 Stage1 Electrical Tracks V10 Opt1

Behavior-preserving performance optimization of the accepted strict, voxel-supported V10 Stage 2 electrical-line reconstruction.

This is a Stage2-only experiment. It consumes saved Stage1 inference artifacts, does not rerun Stage1, does not run Stage3, and leaves production `v4/` unchanged. This directory contains no ONNX export path.

The accepted reference implementation and its reconstruction contract are in [`../v4_stage2_stage1_electrical_v10/`](../v4_stage2_stage1_electrical_v10/). Opt1 must remain quality-equivalent to that reference implementation.

## Reconstruction method

Stage 2 resolves the deployed Stage1 sparse predictions and reconstructs electrical lines only from voxels assigned class `2`. It preserves the accepted production pole extraction.

For each slice, reconstruction:

1. Selects the Stage1 class-2 line voxels.
2. Partitions them into deterministic 26-neighbor connected components.
3. Converts each component into a voxel-adjacency graph.
4. Extracts ordered graph-diameter paths and splits them at sharp turns.
5. Partitions branch voxels so every inferred line voxel belongs to exactly one emitted track.
6. Simplifies path vertices only when sampled chords remain entirely inside the inferred Stage1 line-voxel support.
7. Attaches a line endpoint to a pole only when inferred line and pole voxels make direct contact.
8. Audits voxel identity, assignment completeness, topology and geometry support.

Disconnected Stage1 line components are never bridged, synthetic line voxels are never introduced, and simplified geometry must have a Stage1 voxel-support fraction of `1.0`. Separate conductors remain separate components, including when they independently contact the same pole.

## Opt1 changes

Opt1 preserves the accepted V10 geometry and output contract while reducing work that cannot affect a valid output:

- **Remove prohibited pair enumeration.** The strict model forbids every disconnected-component bridge, so Opt1 does not enumerate all fragment pairs merely to reject them.
- **Avoid unused fragment descriptors.** Descriptors needed only by the removed disconnected-bridge path are not constructed or retained.
- **Index pole-contact voxels.** Pole-support voxels are stored in a coordinate hash. Endpoint attachment checks inspect only the bounded neighboring voxel keys instead of scanning the complete pole-support array.
- **Retain accepted production extraction.** Pole/component production processing uses the accepted reference path. An earlier pole-only shortcut was removed after equivalence testing found serialization differences.
- **Add phase timing.** Opt1 reports label resolution, connected components, graph tracing, assignment auditing, geometry construction and total Stage 2 timing separately.

The reported `disconnected_bridge_pair_enumeration_ms` is therefore `0.0` by construction.

## Quality-equivalence contract

`compare_v10_stage2_quality.py` compares every optimized session with the accepted V10 baseline supplied through `QUALITY_BASELINE_ROOT`.

The following line/electrical outputs must be byte-exact:

- `*_lines.csv`
- `*_line_vertices.csv`
- `*_stage1_line_voxels.csv`
- `*_accepted_line_voxels.csv`
- `*_selected_fragment_bridges.csv`
- `*_stage1_electrical_tracks.csv`
- `*_pole_attachments.csv`

Production `*_poles.csv` and `*_components.csv` must preserve their inventory, schema, row identities and values. Numeric cells are compared with absolute and relative tolerances of `1e-9` to allow benign floating-point serialization differences.

Selected reconstruction audit fields must also match the accepted baseline. Timing paths and fields that describe removed diagnostic work are excluded from semantic equivalence because they are expected to change.

`self_test_compare_v10_stage2_quality.py` verifies that the comparator accepts only bounded production serialization differences and still rejects changes to electrical-line geometry or material pole/component values.

## Strict reconstruction invariants

- Runtime line geometry starts only from deployed Stage1 `label == 2` voxels.
- Production pole detections are preserved.
- Disconnected Stage1 line components are never bridged.
- Every inferred line voxel is assigned to exactly one output track.
- Geometry outside Stage1 line-voxel support is forbidden.
- Line-to-pole attachment requires direct inferred line/pole voxel contact.
- Separate conductor tracks remain separate at shared poles.
- No pole-pair enumeration is used.
- No runtime ground truth is used.
- No synthetic line voxels are introduced.
- Open line endpoints are preserved when no supported pole contact exists.

## Main entry points

- `v4_stage2_stage1_electrical_tracks_opt.py` — optimized reconstruction implementation.
- `run_v4_stage2_stage1_electrical_tracks_opt.py` — optimized per-session runner with detailed timing.
- `learn_velasco_stage1_electrical_profile_opt.py` — fixed electrical profile learning from saved Stage1 outputs.
- `validate_v10_voxel_supported_stage2_opt.py` — independent voxel-support validation.
- `compare_v10_stage2_quality.py` — baseline-versus-Opt1 equivalence gate.
- `self_test_stage1_electrical_tracks_opt.py` — reconstruction self-test.
- `self_test_compare_v10_stage2_quality.py` — comparator regression test.
- `summarize_v10_opt_timing.py` — per-session and aggregate timing summary.
- `run_v10_stage2_opt_experiment.sh` — fault-tolerant all-session optimization driver.
- `launch_v10_stage2_opt_experiment.sh` — background launcher.
- `monitor_v10_stage2_opt_experiment.sh` — run monitor.
- `package_v10_stage2_opt_experiment.sh` — gated experiment-result packager.

## Prerequisites

The optimization driver requires:

- the saved Stage1 run and manifests;
- the accepted Stage 2 refiner bundle;
- the deployed calibration;
- the accepted V10 baseline run directory; and
- `STAGE2_ONLY_COMPLETE.txt` under that baseline run.

`QUALITY_BASELINE_ROOT` must point to the accepted baseline and must not equal the new optimized run root.

## Running the experiment

The launcher uses `/home/agni/LATEST_V10_FAULT_TOLERANT_STAGE2_RUN.txt` as the default quality baseline when `QUALITY_BASELINE_ROOT` is not supplied explicitly:

```bash
bash ops/v4_stage2_stage1_electrical_v10_opt/launch_v10_stage2_opt_experiment.sh
```

To select the baseline explicitly:

```bash
QUALITY_BASELINE_ROOT=/absolute/path/to/accepted/fault_tolerant_v10_<UTC> \
  bash ops/v4_stage2_stage1_electrical_v10_opt/launch_v10_stage2_opt_experiment.sh
```

Monitor the latest experiment with:

```bash
bash ops/v4_stage2_stage1_electrical_v10_opt/monitor_v10_stage2_opt_experiment.sh
```

## Output contract

The optimized all-session runner writes a new UTC-stamped run:

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
        │   ├── <SID>.stage2.ok
        │   ├── <SID>.voxel_support_validation.json
        │   └── <SID>.quality_equivalence.json
        ├── FILE_INVENTORY.txt
        ├── RUN_INFO.txt
        ├── STAGE2_TIMING_SESSION_AVERAGES.txt
        ├── session_map.tsv
        ├── PHASE2_STAGE2_OK.txt
        └── STAGE2_ONLY_COMPLETE.txt
```

`EXPECTED_SESSIONS` defaults to `30`. For every accepted session, the driver requires both voxel-support validation and quality equivalence against the corresponding accepted-baseline session.

## Timing output

The per-session timing CSVs include:

- `production_component_ms`
- `production_refiner_parametric_ms`
- `production_stage2_ms`
- `stage1_label_resolve_ms`
- `line_connected_components_ms`
- `line_trace_ms`
- `line_assignment_audit_ms`
- `line_geometry_ms`
- `disconnected_bridge_pair_enumeration_ms`
- `stage1_electrical_track_ms`
- `stage2_total_ms`

`summarize_v10_opt_timing.py` writes `STAGE2_TIMING_SESSION_AVERAGES.txt`, including baseline and optimized mean Stage 2 timing and the resulting compute speedup when both values are available.

## Completion and packaging gates

The run is complete only when:

- all expected sessions have accepted markers;
- there are zero failed-session markers;
- every session has a voxel-support validation report;
- every session has a quality-equivalence report;
- `STAGE2_TIMING_SESSION_AVERAGES.txt` exists; and
- `STAGE2_ONLY_COMPLETE.txt` exists.

After those conditions pass:

```bash
bash ops/v4_stage2_stage1_electrical_v10_opt/package_v10_stage2_opt_experiment.sh
```

The packager creates `v10_stage2_opt_experiment_results_<UTC>.tar.gz`, its SHA-256 file and `/home/agni/LATEST_V10_STAGE2_OPT_ARCHIVE.txt`.

Opt1 should remain an experiment until its equivalence reports and timing results are reviewed. Its acceptance criterion is lower latency with no material reconstruction-quality change.
