# Pole_Line_3D_Recon

Production and experimental code for three-stage pole and power-line reconstruction from voxelized LiDAR slices.

## Active production version: V4

The accepted production implementation is under [`v4/`](v4/).

Production configuration:

- Stage 1: per-slice voxel inference using the accepted V4 `MultiHeadVoxelNet3D` weights.
- Stage 2: per-slice pole and conductor-segment reconstruction from Stage 1 scores.
- Stage 3: rolling multi-slice world reconstruction using only already-acquired slices.
- Stage 3 window: maximum sequence gap 9 at 50 ft per slice interval, i.e. 450 ft maximum physical span and up to 10 observed slice centers in `[S-9, S]`.
- Accepted Stage 1 runtime: `active_gpu`.
- Accepted batch size: `12`.
- Python execution on Nebius is Docker-only.
- Stage 1, Stage 2, and Stage 3 remain independently executable from durable saved stage outputs.

See [`v4/README.md`](v4/README.md) for architecture, acceptance results, runtime details, and Nebius production workflows.

## Full-dataset operations

Operational helpers for running the accepted V4 implementation over every discovered valid session are under [`ops/v4_full_dataset/`](ops/v4_full_dataset/).

Key entry points:

- `run_v4_all_stage1_on_nebius.sh` — Stage 1 only over all discovered sessions.
- `run_v4_all_stage2_on_nebius.sh` — Stage 2 only from saved Stage 1 outputs.
- `run_v4_all_stage3_on_nebius.sh` — Stage 3 only from saved Stage 2 outputs.
- `run_v4_all_reconstruction_on_nebius.sh` — Stage 2 + Stage 3.
- `run_v4_all_in_one_on_nebius.sh` — complete all-data workflow.
- `package_v4_all_data_results_on_nebius.sh` — package metrics, timings, inference CSVs, and reconstruction outputs while excluding NPZ/model artifacts.
- `download_v4_all_data_results_to_mac.sh` — download packaged results to macOS.
- `make_v4_compact_review_from_results_on_mac.sh` — create a compact review archive from a large all-data result package.

See [`ops/v4_full_dataset/README_FULL_DATASET_RUN.md`](ops/v4_full_dataset/README_FULL_DATASET_RUN.md) for the operational runbook.

## Stage 1 inference performance experiments

Production-preserving Stage 1 execution experiments are isolated under
[`ops/v4_stage1_inference_opt2/`](ops/v4_stage1_inference_opt2/). They keep the
accepted checkpoint, calibration, complete model, active-GPU BF16 path, batch
size 12, channels-last layout, patch/core geometry, fusion, thresholds and
serialization unchanged. Candidate promotion requires exact comparison with
the saved production outputs (`SCORE_ATOL=0`); this is an implementation guard,
not evaluation against the incomplete ground-truth labels.

E0 is the accepted control. E2 was rejected after removing detailed CUDA events
exposed an unsafe asynchronous gather-index source lifetime and changed a
production pole score. E3's lifetime safeguard was exact but produced no
repeatable benefit while events remained enabled. E4 remained
exact in two representative runs but saved only 0.37% on average and was rejected
as operationally insignificant. E5's coordinate/input cache was rejected after
its full gate changed a production pole score, despite passing representative
runs. E5b narrowed the change to immutable coordinate values while retaining
production `torch.cat`, channels-last materialization and fresh final inputs.
E5b passed the original failing session, improved matched Stage 1 wall time by
1.32%, reduced feature assembly by 55.3%, completed Nsight Systems profiling and
reproduced saved production outputs exactly across all 3,738 slices in 30
sessions. See the experiment README and E5b acceptance report for details.

E6 removed detailed CUDA-event timing while retaining E3's exact gather-buffer
lifetime safeguard. E7 adds fixed-shape CUDA Graph replay of the unchanged model
and score-fusion operators. E7 passed two exact paired E6 comparisons and the
complete 30-session / 3,738-slice gate: Stage 1 wall time improved from
282.852 ms to 109.138 ms per slice (61.42%, 2.592x), while baseline-comparable
total improved from 379.022 ms to 203.350 ms (46.35%, 1.864x). The production
branch remains unchanged. E8 asynchronous read/infer/write and E9 in-memory
Stage 1 to Stage 2 handoff are separately gated next phases; all durable output
paths and schemas remain unchanged. See the Opt2 README and E6/E7 acceptance
report for the exact contract and profiling evidence.

## Repository layout

```text
Pole_Line_3D_Recon/
├── README.md
├── v4/                        # accepted production V4 implementation
├── ops/
│   ├── v4_full_dataset/       # full-dataset operational tooling
│   └── v4_stage1_inference_opt2/ # production-gated Stage 1 performance work
└── ...                        # legacy/experimental V6.2 files retained for history and research
```

The root-level V6.2 files are retained as historical/experimental work. They are not the active production implementation on the `v4` branch.

## External artifacts intentionally not stored in Git

This repository must not contain raw datasets, bulk generated outputs, Stage 1
NPZ caches, trained model/checkpoint files, calibration bundles, raw Nsight
binary reports or other large runtime artifacts. Small reviewed Markdown,
text and JSON profiling summaries may be retained as documentation.

Typical external locations on Nebius include:

```text
/data/voxel_csv_combined
/outputs/poleline_voxel_run_session_groups/precision_v4/train/precision_best.pt
/outputs/poleline_voxel_run_session_groups/precision_v4/full_val/calibration.json
/workspace/voxel_poleline/outputs
```

## Git hygiene

Before every push:

```bash
git status --short
git diff --cached --stat
git diff --cached --name-only | grep -Ei '\.(npz|pt|pth|ckpt|joblib|onnx|engine|safetensors|csv|csv\.gz)$' || true
git diff --cached --name-only | grep -E '(^|/)(__pycache__|\.DS_Store|\._)' || true
```

Generated data/model artifacts and macOS metadata should not be committed.
