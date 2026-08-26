#!/usr/bin/env bash
# Must run inside the V4 Docker image.
set -euo pipefail
export PYTHONPATH=/workspace/v4
python -m py_compile /workspace/v4_full_ops/*.py
for f in /workspace/v4_full_ops/*.sh; do bash -n "$f"; done
python /workspace/v4_full_ops/export_score_v4_stage1.py --help >/dev/null
python /workspace/v4_full_ops/profile_stage2_session.py --help >/dev/null
python /workspace/v4_full_ops/profile_stage3_session.py --help >/dev/null
python /workspace/v4_full_ops/summarize_v4_all_data.py --help >/dev/null
python /workspace/v4/smoke_test_v4_stage_contracts.py
python /workspace/v4/smoke_test_v4_stage2_runtime.py
python /workspace/v4/smoke_test_v4_stage3_incremental.py
python /workspace/v4/smoke_test_v4_450ft_window.py
python /workspace/v4/smoke_test_v4_discovery_and_errors.py
python /workspace/v4/smoke_test_v4_cli_error_paths.py
echo V4_FULL_DATASET_OPS_DOCKER_VALIDATION_OK
