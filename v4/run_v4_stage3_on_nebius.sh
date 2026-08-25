#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/v4_nebius_common.sh"
v4_print_context
BACKUP=$(v4_backup_code); echo "Code backup: $BACKUP"
NAME=${NAME:-v4-stage3-$(printf '%s' "$V4_SESSION_SAFE" | tr -cd 'A-Za-z0-9_.-' | cut -c1-70)}
CMD="python run_v4_stage3.py --stage2_dir '$V4_RUN_ROOT' --output_dir '$V4_RUN_ROOT/stage3_independent' --session_filter '$V4_SESSION_FILTER' --max_sequence_gap 9 --slice_length_ft 50 --max_span_length_ft 450 --latest_only '${LATEST_ONLY:-0}' --resume '${RESUME:-1}'"
v4_detached_run "$NAME" "$V4_RUN_ROOT/stage_logs/stage3.log" "$CMD"
