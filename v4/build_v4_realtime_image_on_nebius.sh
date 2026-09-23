#!/usr/bin/env bash
# Disconnect-safe Docker image build. Default invocation backgrounds the build with nohup.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IMAGE=${IMAGE:-va-v4-realtime:torch241-cu121}
STATE_DIR="$HERE/.v4_nebius_state"
mkdir -p "$STATE_DIR"

if [[ "${V4_BUILD_FOREGROUND:-0}" != 1 ]]; then
  STAMP=$(date -u +%Y%m%dT%H%M%SZ)
  LOG="$STATE_DIR/build_${STAMP}.log"
  nohup env V4_BUILD_FOREGROUND=1 IMAGE="$IMAGE" "$0" >"$LOG" 2>&1 </dev/null &
  PID=$!
  printf '%s\n' "$PID" > "$STATE_DIR/build_pid.txt"
  printf '%s\n' "$LOG" > "$STATE_DIR/build_log.txt"
  echo "Started background V4 Docker build pid=$PID"
  echo "Persistent log: $LOG"
  echo "Check after reconnect: $HERE/show_v4_build_state_on_nebius.sh"
  exit 0
fi

if docker info >/dev/null 2>&1; then DOCKER=(docker); else DOCKER=(sudo docker); fi
{
  echo "BUILD_STARTED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "IMAGE=$IMAGE"
  "${DOCKER[@]}" build -f "$HERE/Dockerfile.v4_realtime" -t "$IMAGE" "$HERE"
  "${DOCKER[@]}" run --rm --gpus all "$IMAGE" python - <<'PY'
import torch, scipy, sklearn, pandas, numpy
assert torch.cuda.is_available(), 'CUDA is not available in the built image'
print('torch=',torch.__version__,'cuda=',torch.version.cuda,'cuda_available=',torch.cuda.is_available())
print('gpu=',torch.cuda.get_device_name(0),'bf16=',torch.cuda.is_bf16_supported())
print('scipy=',scipy.__version__,'sklearn=',sklearn.__version__,'pandas=',pandas.__version__,'numpy=',numpy.__version__)
PY
  echo "V4_DOCKER_IMAGE_BUILD_OK image=$IMAGE completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 
