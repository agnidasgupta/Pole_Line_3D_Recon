#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/v4_nebius_common.sh"
v4_print_context
v4_require_acceptance
BACKUP=$(v4_backup_code)
echo "Code backup: $BACKUP"
NAME=${NAME:-v4-prod-$(printf '%s' "$V4_SESSION_SAFE" | tr -cd 'A-Za-z0-9_.-' | cut -c1-80)}
LOG_CONTAINER="$V4_RUN_ROOT/production_runner.log"
CMD="python run_v4_realtime_session.py \
 --input_dir /data/voxel_csv_combined --session_filter '$V4_SESSION_FILTER' --output_dir '$V4_RUN_ROOT' \
 --model_path '$V4_MODEL' --calibration_json '$V4_CAL' --stage2_bundle '$V4_STAGE2' \
 --batch_size '$V4_BATCH_SIZE' --amp '${AMP:-bf16}' --compile_model '${COMPILE_MODEL:-0}' \
 --evaluate_all_cores '$V4_EVALUATE_ALL_CORES' --gpu_coord_channels '$V4_GPU_COORD_CHANNELS' --fixed_batch_shape '$V4_FIXED_BATCH_SHAPE' \
 --max_sequence_gap 9 --slice_length_ft 50 --max_span_length_ft 450 \
 --stage3_every_slice 1 --stage3_execution inprocess --stage3_inmemory_cache 1 \
 --write_row_csv '${WRITE_ROW_CSV:-0}' --resume '${RESUME:-1}' --max_slices '${MAX_SLICES:-0}'"
v4_detached_run "$NAME" "$LOG_CONTAINER" "$CMD"
