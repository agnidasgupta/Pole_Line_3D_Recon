#!/usr/bin/env bash
# Host launcher for Docker-only full-dataset ops validation.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/v4_full_ops_common.sh"
v4_require_acceptance
ROOT="$V4_PROD_ROOT_HOST/full_dataset_preflight"; mkdir -p "$ROOT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ); LOG_HOST="$ROOT/preflight_${STAMP}.log"; LOG=$(v4_to_container_output "$LOG_HOST")
NAME="v4-fullops-preflight-${V4_DEPLOY_SHORT}-${STAMP}"; NAME=$(printf '%s' "$NAME" | cut -c1-120)
CID=$("${V4_DOCKER[@]}" run -d --name "$NAME" --gpus all \
  -e PYTHONPYCACHEPREFIX=/tmp/pycache -e MPLCONFIGDIR=/tmp/matplotlib -e XDG_CACHE_HOME=/tmp/xdg-cache -e TMPDIR=/tmp \
  --mount "type=bind,source=$V4_ACCEPTED_DIR,target=/workspace/v4,readonly" \
  --mount "type=bind,source=$HERE,target=/workspace/v4_full_ops,readonly" \
  --mount "type=bind,source=$V4_HOST_OUTPUTS,target=/outputs" \
  --mount "type=bind,source=$V4_HOST_INPUT,target=/data/voxel_csv_combined,readonly" \
  --workdir /workspace/v4 "$V4_IMAGE" bash -lc "set -euo pipefail; bash /workspace/v4_full_ops/validate_v4_full_ops_inside_docker.sh 2>&1 | tee '$LOG'")
printf '%s\n' "$NAME" > "$ROOT/LATEST_PREFLIGHT_CONTAINER.txt"; printf '%s\n' "$LOG_HOST" > "$ROOT/LATEST_PREFLIGHT_LOG.txt"
echo "Started full-data ops preflight: $NAME ($CID)"
echo "Log: $LOG_HOST"
echo "Check after reconnect:"
echo "  docker logs $NAME --tail 100"
