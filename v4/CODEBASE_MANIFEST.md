# V4 realtime production candidate — codebase manifest

This package contains source only. It does not contain trained checkpoints, Stage-2 joblib bundles, raw LiDAR CSVs, NPZ datasets, or experiment outputs.

## V4 Stage 1 — one slice only

- `v4_realtime_core.py` — V4 model loading, calibration, sparse raw-slice preparation, full/active core scheduling, local coordinate channels, CUDA-synchronized Stage-1 timing.
- `v4_realtime_pipeline.py` — persistent Stage-1 + Stage-2 one-slice processor.
- `voxel_common.py` — accepted V4 network classes/utilities.
- `precision_common.py` — V4 score/calibration utilities.
- `compare_v4_runtime_variants.py` — four-way real-data Stage-1 equivalence/speed gate.
- `select_v4_runtime_mode.py` — conservative production selector (`active_cpu` only when safe, otherwise `full_cpu`).
- `benchmark_v4_stage1_batch_sizes.py` — batch-size equivalence/speed sweep.
- `run_v4_runtime_variant_gate_on_nebius.sh`
- `run_v4_stage1_batch_sweep_on_nebius.sh`

## V4 Stage 2 — one slice only

- `v4_sparse_components.py` — exact sparse 26-neighbor component extraction.
- `v4_stage2_local.py` — local component feature schema and target logic.
- `v4_stage2_runtime.py` — learned local refiner inference and pole/line parameterization.
- `mine_v4_stage2_components.py` — offline current-slice component mining for Stage-2 training.
- `train_v4_stage2_refiners.py` — ExtraTrees pole/line refiners and threshold selection.
- `run_v4_stage2_training.sh`
- `run_v4_stage2_training_on_nebius.sh`

## V4 Stage 3 — rolling past-only multi-slice reconstruction

- `reconstruct_v4_stage3.py` — deterministic pole/conductor reconstruction. With `--latest_slice S`, it reads only sequence rows `S-9 ... S` and never future rows.
- `run_v4_realtime_session.py` — persistent realtime session runner: Stage 1 current slice -> Stage 2 current slice -> Stage 3 rolling update after every slice.
- `run_v4_realtime_session_on_nebius.sh`

The production runner rejects all-session reconstruction and enforces 9 increments / 450 ft.

## Timing / validation / diagnostics

- `run_v4_realtime_benchmark_on_nebius.sh` — clean short/full-session realtime timing replay.
- `summarize_v4_realtime_timing.py` — P50/P95 and Stage-3 window-size timing summary.
- `verify_v4_realtime_replay.py` — past-only/no-future Stage-3 verification and output integrity checks.
- `verify_v4_realtime_replay_on_nebius.sh`
- `collect_v4_realtime_diagnostics.py`
- `validate_v4_production_source_contract.py`
- `smoke_test_v4_realtime.py`
- `smoke_test_v4_stage2_runtime.py`
- `smoke_test_v4_stage3_incremental.py`
- `run_v4_realtime_preflight_on_nebius.sh`
- `package_v4_production_review_on_nebius.sh`
- `package_v4_realtime_diagnostics_on_nebius.sh`
- `download_v4_production_results_to_mac.sh`
- `download_v4_realtime_results_to_mac.sh`
- `profile_environment.py`

## Environment / deployment

- `Dockerfile.v4_realtime`
- `requirements.txt`
- `build_v4_realtime_image_on_nebius.sh`
- `.gitignore`

## Documentation

- `README.md`
- `NEBIUS_V4_REALTIME_RUNBOOK.md`
- `GITHUB_V4_REALTIME_INSTRUCTIONS.md`
- `V4_REALTIME_PERFORMANCE_REVIEW.md`
- `README_STEP11_ACTIVE_CORE_CPU_COORD.md`
- `CODEBASE_MANIFEST.md`
- `SHA256SUMS.txt`
