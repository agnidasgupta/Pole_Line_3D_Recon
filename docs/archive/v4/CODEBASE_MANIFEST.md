# V4 production test codebase manifest

## Stage 1 — independently executable

- `run_v4_stage1.py` — raw slice -> durable Stage 1 artifact/manifest.
- `run_v4_stage1_on_nebius.sh` — detached Docker-only Nebius launcher.
- `v4_realtime_core.py` — model/calibration, sparse input, reference and optimized GPU inference paths, fine-grained Stage 1 timings.
- `precision_common.py`, `voxel_common.py` — accepted V4 model/scoring utilities.
- `compare_v4_runtime_variants.py` — H100 runtime gate including downstream Stage 2 equivalence.
- `benchmark_v4_stage1_batch_sizes.py`, `select_v4_batch_size.py` — equivalent-batch latency selection.
- `select_v4_runtime_mode.py`, `validate_v4_runtime_gate.py` — production runtime promotion.

## Stage 2 — independently executable

- `run_v4_stage2.py` — durable Stage 1 -> poles/line segments/vertices + durable manifest.
- `run_v4_stage2_on_nebius.sh` — detached Docker-only Nebius launcher.
- `v4_sparse_components.py` — sparse connected components.
- `v4_stage2_local.py`, `v4_stage2_runtime.py` — feature/refiner/parametric reconstruction.
- `mine_v4_stage2_components.py`, `train_v4_stage2_refiners.py` — offline next-training-run utilities.
- `run_v4_stage2_training.sh` — container-internal training body; refuses host execution.
- `run_v4_stage2_training_on_nebius.sh` — Docker launcher for future retraining.

## Stage 3 — independently executable

- `run_v4_stage3.py` — durable Stage 2 -> rolling Stage 3 snapshots.
- `run_v4_stage3_on_nebius.sh` — detached Docker-only Nebius launcher.
- `reconstruct_v4_stage3.py` — 9-sequence-gap / 450-ft reconstruction, cached rolling Stage 2 frames, fragment joining, span completion, hidden poles, attachments and side lines.

## Realtime orchestration and durable contracts

- `run_v4_realtime_session.py` — Stage1 -> Stage2 -> Stage3 realtime path, RAM handoff plus atomic durable boundaries and recovery.
- `run_v4_production_on_nebius.sh` — accepted-deployment production launcher.
- `v4_realtime_pipeline.py` — persistent Stage 1/2 objects.
- `v4_stage_contracts.py` — atomic artifact/manifest functions and stable stage paths.

## Nebius deployment/persistence

- `v4_nebius_common.sh` — shell-only path/session/artifact discovery, fingerprints, persistent roots, Docker helpers.
- `v4_code_fingerprint.sh` — deterministic code fingerprint.
- `build_v4_realtime_image_on_nebius.sh`, `show_v4_build_state_on_nebius.sh`, `Dockerfile.v4_realtime`, `requirements.txt`.
- `v4_preflight_inside_docker.sh`, `run_v4_realtime_preflight_on_nebius.sh` — detached Docker preflight with durable success/failure markers.
- `run_v4_production_tests_on_nebius.sh` — detached acceptance launcher.
- `v4_production_acceptance_inside_docker.sh` — complete H100 acceptance sequence.
- `show_v4_production_state_on_nebius.sh` — reconnect/status helper.
- `package_v4_production_state_on_nebius.sh` — persistent state backup.

## Validation and diagnostics

- `v4_code_validation_inside_docker.sh` — compiles/imports/parses all Python and runs synthetic tests.
- `validate_v4_production_source_contract.py`.
- `smoke_test_v4_realtime.py`.
- `smoke_test_v4_stage2_runtime.py`.
- `smoke_test_v4_stage3_incremental.py`.
- `smoke_test_v4_stage_contracts.py`.
- `smoke_test_v4_450ft_window.py`.
- `smoke_test_v4_discovery_and_errors.py`.
- `smoke_test_v4_cli_error_paths.py`.
- `smoke_test_v4_nebius_discovery.sh` — shell-only host discovery test.
- `verify_v4_realtime_replay.py` / `verify_v4_realtime_replay_on_nebius.sh`.
- `summarize_v4_realtime_timing.py`.
- `collect_v4_realtime_diagnostics.py`.
- `package_v4_review_bundle_on_nebius.sh`.
- `download_v4_review_bundle_to_mac.sh` plus compatibility aliases.

## Documentation

- `README.md`
- `NEBIUS_V4_REALTIME_RUNBOOK.md`
- `PRODUCTION_ACCEPTANCE_CHECKLIST.md`
- `GITHUB_V4_REALTIME_INSTRUCTIONS.md`
- `V4_REALTIME_PERFORMANCE_REVIEW.md`
- `CODEBASE_MANIFEST.md`
- `LOCAL_VALIDATION_REPORT.txt`
- `SHA256SUMS.txt`
