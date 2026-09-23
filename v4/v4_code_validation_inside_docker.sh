#!/usr/bin/env bash
# Comprehensive Python/source validation. Must run inside the V4 Docker image.
set -euo pipefail
[[ -f /.dockerenv ]] || { echo 'ERROR: v4_code_validation_inside_docker.sh must run inside Docker.' >&2; exit 2; }
cd "$(dirname "${BASH_SOURCE[0]}")"

echo '[validation] compile every Python source'
python -m compileall -q .

echo '[validation] import every library module'
python - <<'PY'
mods = [
    'precision_common','voxel_common','v4_realtime_core','v4_realtime_pipeline',
    'v4_sparse_components','v4_stage2_local','v4_stage2_runtime','v4_stage_contracts',
]
for m in mods:
    __import__(m)
print('V4_LIBRARY_IMPORTS_OK', len(mods))
PY

echo '[validation] parse every command-line entry point'
for f in \
  benchmark_v4_stage1_batch_sizes.py collect_v4_realtime_diagnostics.py \
  compare_v4_runtime_variants.py mine_v4_stage2_components.py \
  reconstruct_v4_stage3.py run_v4_realtime_session.py run_v4_stage1.py \
  run_v4_stage2.py run_v4_stage3.py select_v4_batch_size.py \
  select_v4_runtime_mode.py summarize_v4_realtime_timing.py \
  train_v4_stage2_refiners.py validate_v4_runtime_gate.py \
  verify_v4_realtime_replay.py; do
  python "$f" --help >/dev/null
done
TMP_PROFILE=$(mktemp)
python profile_environment.py "$TMP_PROFILE" >/dev/null
test -s "$TMP_PROFILE"
rm -f "$TMP_PROFILE"

echo '[validation] source contracts + synthetic runtime/error paths'
python validate_v4_production_source_contract.py
python smoke_test_v4_realtime.py
python smoke_test_v4_stage2_runtime.py
python smoke_test_v4_stage3_incremental.py
python smoke_test_v4_stage_contracts.py
python smoke_test_v4_450ft_window.py
python smoke_test_v4_discovery_and_errors.py
python smoke_test_v4_cli_error_paths.py
python smoke_test_v4_replay_verifier_contract.py

echo 'V4_CODE_VALIDATION_INSIDE_DOCKER_OK'
