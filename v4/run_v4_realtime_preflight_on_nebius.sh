#!/usr/bin/env bash
# Disconnect-safe preflight. Host runs shell-only discovery; Python runs in detached Docker.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
"$HERE/smoke_test_v4_nebius_discovery.sh"
source "$HERE/v4_nebius_common.sh"
v4_print_context
BACKUP=$(v4_backup_code); echo "Code backup: $BACKUP"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
PREFLIGHT_HOST="$V4_RUN_ROOT_HOST/preflight/$STAMP"
mkdir -p "$PREFLIGHT_HOST"
PREFLIGHT=$(v4_to_container_output "$PREFLIGHT_HOST")
NAME=${NAME:-v4-preflight-$(printf '%s' "$V4_SESSION_SAFE" | tr -cd 'A-Za-z0-9_.-' | cut -c1-40)-${V4_DEPLOY_SHORT:0:8}}
CMD="bash v4_preflight_inside_docker.sh '$PREFLIGHT' '$V4_MODEL' '$V4_CAL' '$V4_STAGE2'"
v4_detached_run "$NAME" "$PREFLIGHT/preflight.log" "$CMD"
printf '%s\n' "$PREFLIGHT_HOST" > "$V4_RUN_ROOT_HOST/LATEST_PREFLIGHT_ROOT.txt"
echo "Preflight root: $PREFLIGHT_HOST"
