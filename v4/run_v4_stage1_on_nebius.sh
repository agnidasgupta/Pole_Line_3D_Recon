#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/v4_nebius_common.sh"
v4_print_context
v4_require_current_gate
BACKUP=$(v4_backup_code); echo "Code backup: $BACKUP"
NAME=${NAME:-v4-stage1-$(printf '%s' "$V4_SESSION_SAFE" | tr -cd 'A-Za-z0-9_.-' | cut -c1-70)}
CMD="python run_v4_stage1.py --input_dir /data/voxel_csv_combined --session_filter '$V4_SESSION_FILTER' --output_dir '$V4_RUN_ROOT' --model_path '$V4_MODEL' --calibration_json '$V4_CAL' --batch_size '$V4_BATCH_SIZE' --amp '${AMP:-bf16}' --compile_model '${COMPILE_MODEL:-0}' --evaluate_all_cores '$V4_EVALUATE_ALL_CORES' --gpu_coord_channels '$V4_GPU_COORD_CHANNELS' --fixed_batch_shape '$V4_FIXED_BATCH_SHAPE' --resume '${RESUME:-1}' --max_slices '${MAX_SLICES:-0}'"
v4_detached_run "$NAME" "$V4_RUN_ROOT/stage_logs/stage1.log" "$CMD"
