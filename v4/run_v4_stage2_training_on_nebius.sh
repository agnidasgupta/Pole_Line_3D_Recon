#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
HOST_OUTPUTS=${HOST_OUTPUTS:-/workspace/voxel_poleline/outputs}
RUNTIME_MODE_ENV=${RUNTIME_MODE_ENV:-$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_realtime/diagnostics/v4_runtime_mode.env}
if [[ -f "$RUNTIME_MODE_ENV" && -z ${EVALUATE_ALL_CORES+x} && -z ${GPU_COORD_CHANNELS+x} ]]; then
  # Use exactly the H100-gated Stage-1 path for Stage-2 mining and realtime replay.
  source "$RUNTIME_MODE_ENV"
  echo "Using gated V4 runtime: ${V4_RUNTIME_MODE:-unknown} (EVALUATE_ALL_CORES=$EVALUATE_ALL_CORES GPU_COORD_CHANNELS=$GPU_COORD_CHANNELS)"
else
  EVALUATE_ALL_CORES=${EVALUATE_ALL_CORES:-1}
  GPU_COORD_CHANNELS=${GPU_COORD_CHANNELS:-0}
  echo "Using explicit/safe V4 runtime (EVALUATE_ALL_CORES=$EVALUATE_ALL_CORES GPU_COORD_CHANNELS=$GPU_COORD_CHANNELS)"
fi
IMAGE=${IMAGE:-va-v4-realtime:torch241-cu121}
NAME=${NAME:-poleline-v4-stage2-training}
sudo docker rm -f "$NAME" >/dev/null 2>&1 || true
# V4_REALTIME_WRITABLE_CACHE_ENV_V2
sudo docker run -d --name "$NAME" --gpus all \
  -e PYTHONPYCACHEPREFIX=/tmp/pycache \
  -e MPLCONFIGDIR=/tmp/matplotlib \
  -e XDG_CACHE_HOME=/tmp/xdg-cache \
  -e TORCH_HOME=/tmp/torch-home \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
  -e TRITON_CACHE_DIR=/tmp/triton \
  -e CUDA_CACHE_PATH=/tmp/cuda-cache \
  -e NUMBA_CACHE_DIR=/tmp/numba-cache \
  -e JOBLIB_TEMP_FOLDER=/tmp/joblib \
  -e TMPDIR=/tmp \
  --mount type=bind,source="$HERE",target=/workspace/voxel_poleline,readonly \
  --mount type=bind,source="$HOST_OUTPUTS",target=/outputs \
  --workdir /workspace/voxel_poleline \
  -e DATASET_DIR="${DATASET_DIR:-/outputs/poleline_voxel_run_session_groups/dataset_hardneg_v4opt_uncompressed}" \
  -e OUT="${OUT:-/outputs/poleline_voxel_run_session_groups/v4_realtime}" \
  -e MODEL="${MODEL:-/outputs/poleline_voxel_run_session_groups/precision_v4/train/precision_best.pt}" \
  -e CAL="${CAL:-/outputs/poleline_voxel_run_session_groups/precision_v4/full_val/calibration.json}" \
  -e BATCH_SIZE="${BATCH_SIZE:-12}" -e AMP="${AMP:-bf16}" -e COMPILE_MODEL="${COMPILE_MODEL:-0}" -e EVALUATE_ALL_CORES="${EVALUATE_ALL_CORES:-1}" -e GPU_COORD_CHANNELS="${GPU_COORD_CHANNELS:-0}" -e RESUME="${RESUME:-1}" \
  bash ./run_v4_stage2_training.sh
printf 'Started %s\nFollow: sudo docker logs -f %s\nWait: sudo docker wait %s\n' "$NAME" "$NAME" "$NAME"
