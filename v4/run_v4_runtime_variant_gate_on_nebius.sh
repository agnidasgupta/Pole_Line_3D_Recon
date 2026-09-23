#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/v4_nebius_common.sh"
v4_print_context
BACKUP=$(v4_backup_code); echo "Code backup: $BACKUP"
NAME=${NAME:-v4-gate-$(printf '%s' "$V4_SESSION_SAFE" | tr -cd 'A-Za-z0-9_.-' | cut -c1-48)-${V4_DEPLOY_SHORT:0:8}}
GATE="$V4_DIAG_ROOT/runtime_variant_equivalence.json"
RUNTIME_ENV="$V4_DIAG_ROOT/v4_runtime_mode.env"
SWEEP="$V4_DIAG_ROOT/batch_sweep"
BATCH_ENV="$V4_DIAG_ROOT/v4_batch_size.env"
CMD="mkdir -p '$V4_DIAG_ROOT' '$SWEEP'; \
 python compare_v4_runtime_variants.py --input_dir /data/voxel_csv_combined --max_files '${MAX_FILES:-32}' --model_path '$V4_MODEL' --calibration_json '$V4_CAL' --stage2_bundle '$V4_STAGE2' --batch_size '${GATE_BATCH_SIZE:-12}' --amp '${AMP:-bf16}' --score_tol '${SCORE_TOL:-1e-4}' --output '$GATE'; \
 python validate_v4_runtime_gate.py --gate_json '$GATE'; \
 python select_v4_runtime_mode.py --gate_json '$GATE' --output_env '$RUNTIME_ENV'; \
 source '$RUNTIME_ENV'; \
 python benchmark_v4_stage1_batch_sizes.py --input_dir /data/voxel_csv_combined --model_path '$V4_MODEL' --calibration_json '$V4_CAL' --stage2_bundle '$V4_STAGE2' --runtime_mode \"\$V4_RUNTIME_MODE\" --batch_sizes '${BATCH_SIZES:-8,12,16,20,24,32}' --reference_batch '${REFERENCE_BATCH:-12}' --max_files '${BATCH_SWEEP_FILES:-8}' --amp '${AMP:-bf16}' --score_tol '${SCORE_TOL:-1e-4}' --output_dir '$SWEEP'; \
 python select_v4_batch_size.py --summary_json '$SWEEP/batch_size_summary.json' --output_env '$BATCH_ENV'; \
 printf '%s\\n' '$V4_DEPLOY_FINGERPRINT' > '$V4_DIAG_ROOT/gated_deployment_sha256.txt'; \
 echo V4_RUNTIME_TUNING_OK; cat '$RUNTIME_ENV'; cat '$BATCH_ENV'"
v4_detached_run "$NAME" "$V4_DIAG_ROOT/runtime_tuning.log" "$CMD"
