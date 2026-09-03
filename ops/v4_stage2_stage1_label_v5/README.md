# V4 Stage 2 — deployed Stage-1 label preservation experiment (V5)

This experiment addresses a specific failure mode: V4 Stage 1 has already
classified occupied voxels as line, but V4 Stage 2 merges, rejects, or
parameterizes them in a way that loses one or more parallel conductor lanes.

## Runtime contract

- Stage 1 is not rerun or modified.
- Stage 3 is not run.
- The deployed V4 Stage-1 decision is reproduced by
  `v4_realtime_core.label_from_scores()` using the accepted calibration.
- Every emitted Stage-2 conductor voxel is an occupied coordinate whose deployed
  Stage-1 label is exactly class `2`.
- No pole pair is enumerated, no empty voxel is filled, and no synthetic Stage-2
  line is created.
- Ground truth is used only by the offline profile selector. The selected
  runtime processor does not read GT.

## Stage-2 correction

1. Preserve production V4 pole extraction.
2. Rebuild line support from the exact deployed Stage-1 line mask.
3. Find exact 26-connected components.
4. Decompose merged parallel conductors by persistent cross-section modes.
   Short diagonal/vegetation bridges are prevented from chaining modes because
   only cross-section bins persistent across many longitudinal positions may
   define a lane.
5. Split each lane at unsupported longitudinal gaps.
6. Apply strict line geometry and the production physical/refiner gates.
7. Permit a tightly constrained, exact-label-backed geometry override when the
   ExtraTrees refiner rejects an exceptionally strong lane.
8. Write standard V4 Stage-2 outputs plus voxel/component audit files.

## Offline automatic selection

The selector compares production Stage 2 with several decomposition profiles.
For `VELASCO_CUT_CP/session1`:

- slices 0–20 are a strict precision-control window;
- slices 21–40 are the recall-recovery window.

A profile must recover correct deployed Stage-1 line voxels/components without
increasing false Stage-1 line voxels, false components, unsupported span length,
or cross-session GT guardrail errors. If none passes, the run stops and packages
selection diagnostics rather than applying an unsafe profile.

## Canonical output structure

The directory structure is intentionally unchanged from prior canonical
Stage-2/Stage-3 quality runs:

```text
/workspace/voxel_poleline/outputs/
└── poleline_voxel_run_session_groups/
    └── v4_stage23_quality/
        └── full_run_v2_<UTC>/
            ├── stage2/<SID>/
            ├── stage3/                  # reserved, empty
            ├── selection/
            ├── logs/stage2/
            ├── logs/stage3/             # reserved
            ├── timing/stage2/
            ├── status/
            ├── RUN_INFO.txt
            ├── session_map.tsv
            ├── PHASE2_STAGE2_OK.txt
            └── STAGE2_ONLY_COMPLETE.txt
```

## Files

- `v4_stage2_stage1_label.py`: experimental Stage-2 processor.
- `select_v4_stage2_stage1_label_profile.py`: GT-only offline selector.
- `run_v4_stage2_stage1_label.py`: one-session Stage-2 runner preserving the
  production output contract.
- `self_test_stage1_label.py`: deterministic lane-decomposition/runtime tests.
- `run_v4_stage2_stage1_label_v5_on_nebius.sh`: preflight, automatic selection,
  all-session Stage 2, validation, packaging.
- `monitor_v4_stage2_stage1_label_v5_on_nebius.sh`: reconnect-safe status.
- `download_v4_stage2_stage1_label_v5_to_mac.sh`: download, checksum, extraction.
- `install_v4_stage2_stage1_label_v5_bundle_on_nebius.sh`: authentication-free
  installation/update from a Git bundle transferred from the Mac.
