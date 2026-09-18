#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/v4_full_ops_common.sh"
ROOT="$V4_PROD_ROOT_HOST/full_dataset_preflight"
NAME=$(cat "$ROOT/LATEST_PREFLIGHT_CONTAINER.txt" 2>/dev/null || true)
LOG=$(cat "$ROOT/LATEST_PREFLIGHT_LOG.txt" 2>/dev/null || true)
[[ -n "$NAME" ]] || { echo "No full-data ops preflight container found."; exit 1; }
"${V4_DOCKER[@]}" ps -a --filter "name=^/${NAME}$" --format 'table {{.Names}}\t{{.Status}}\t{{.ID}}' || true
[[ -n "$LOG" && -f "$LOG" ]] && { echo "--- preflight log ---"; tail -120 "$LOG"; }
