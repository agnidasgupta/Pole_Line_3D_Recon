# Codebase Manifest — `v4` branch

## Active production code

`v4/` contains the accepted V4 realtime implementation.

Required independent stage entry points:

- `v4/run_v4_stage1.py`
- `v4/run_v4_stage2.py`
- `v4/run_v4_stage3.py`
- `v4/run_v4_stage1_on_nebius.sh`
- `v4/run_v4_stage2_on_nebius.sh`
- `v4/run_v4_stage3_on_nebius.sh`

Core production modules include:

- `v4/v4_realtime_core.py`
- `v4/v4_realtime_pipeline.py`
- `v4/v4_stage2_runtime.py`
- `v4/reconstruct_v4_stage3.py`
- `v4/v4_stage_contracts.py`
- `v4/v4_nebius_common.sh`
- `v4/v4_code_fingerprint.sh`

Production validation/recovery includes the runtime gate, batch-size gate, replay verifier, production source-contract validator, smoke tests, review packager, and acceptance/recovery launchers under `v4/`.

## Full-dataset operations

`ops/v4_full_dataset/` contains operational tooling that calls the accepted `v4/` implementation without changing its production fingerprint.

Required all-data stage entry points:

- `ops/v4_full_dataset/run_v4_all_stage1_on_nebius.sh`
- `ops/v4_full_dataset/run_v4_all_stage2_on_nebius.sh`
- `ops/v4_full_dataset/run_v4_all_stage3_on_nebius.sh`
- `ops/v4_full_dataset/run_v4_all_reconstruction_on_nebius.sh`
- `ops/v4_full_dataset/run_v4_all_in_one_on_nebius.sh`

Supporting operations include Docker-only validation, reporting, metrics/timing aggregation, packaging, Mac download, and compact-review creation.

## Production contract

- Stage 1: newly arrived slice only.
- Stage 2: same slice only, using Stage 1 results.
- Stage 3: current plus previously acquired slices only.
- Maximum Stage 3 sequence gap: 9.
- Slice interval: 50 ft.
- Maximum span: 450 ft.
- Maximum observed slice centers in a full legal window: 10 (`S-9` through `S`).
- Accepted Stage 1 runtime: `active_gpu`.
- Accepted Stage 1 batch size: 12.
- Nebius Python execution: Docker-only.
- All stages preserve durable boundaries so later stages can be rerun independently.

## Legacy code

Root-level V6.2 files are retained for historical/experimental work. They are not the active production implementation on the `v4` branch.

## Files intentionally excluded from Git

Do not commit:

- raw datasets;
- generated inference/reconstruction outputs;
- `.npz` stage caches;
- model/checkpoint files (`.pt`, `.pth`, `.ckpt`, `.onnx`, `.engine`, `.safetensors`, `.joblib`);
- local environments;
- `__pycache__`, `.pyc`, `.DS_Store`, or AppleDouble `._*` metadata;
- large result archives.
