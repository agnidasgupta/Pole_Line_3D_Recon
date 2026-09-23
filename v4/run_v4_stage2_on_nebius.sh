#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/v4_nebius_common.sh"
v4_print_context
BACKUP=$(v4_backup_code); echo "Code backup: $BACKUP"
NAME=${NAME:-v4-stage2-$(printf '%s' "$V4_SESSION_SAFE" | tr -cd 'A-Za-z0-9_.-' | cut -c1-70)}
CMD="python run_v4_stage2.py --stage1_dir '$V4_RUN_ROOT' --output_dir '$V4_RUN_ROOT' --session_filter '$V4_SESSION_FILTER' --stage2_bundle '$V4_STAGE2' --resume '${RESUME:-1}' --max_slices '${MAX_SLICES:-0}'"
v4_detached_run "$NAME" "$V4_RUN_ROOT/stage_logs/stage2.log" "$CMD"
