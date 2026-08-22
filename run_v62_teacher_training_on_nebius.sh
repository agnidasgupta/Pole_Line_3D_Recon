#!/usr/bin/env bash
set -euo pipefail
IMAGE="${IMAGE:-va-voxel-poleline:v6.2-three-stage}"
NAME="${NAME:-poleline-v62-teacher-training}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HOST_CODE="${HOST_CODE:-$SCRIPT_DIR}"
HOST_OUTPUTS="${HOST_OUTPUTS:-/workspace/voxel_poleline/outputs}"
HOST_RAW_INPUT_DIR="${HOST_RAW_INPUT_DIR:-/data/voxel_csv_combined}"

test -d "$HOST_CODE" || { echo "Code directory missing: $HOST_CODE"; exit 2; }
test -d "$HOST_OUTPUTS" || { echo "Outputs directory missing: $HOST_OUTPUTS"; exit 2; }
test -d "$HOST_RAW_INPUT_DIR" || { echo "Raw input directory missing: $HOST_RAW_INPUT_DIR"; exit 2; }

sudo docker rm -f "$NAME" >/dev/null 2>&1 || true
sudo docker run -d --name "$NAME" --gpus all --shm-size=64g \
  --mount type=bind,source="$HOST_CODE",target=/workspace/voxel_poleline,readonly \
  --mount type=bind,source="$HOST_OUTPUTS",target=/outputs \
  --mount type=bind,source="$HOST_RAW_INPUT_DIR",target=/data/voxel_csv_combined,readonly \
  -e PYTHONPYCACHEPREFIX=/tmp/pycache \
  -e EPOCHS="${EPOCHS:-30}" -e SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-18000}" -e EVAL_SAMPLES="${EVAL_SAMPLES:-3072}" \
  -e TRAIN_BATCH="${TRAIN_BATCH:-3}" -e GRAD_ACCUM="${GRAD_ACCUM:-2}" -e EVAL_BATCH="${EVAL_BATCH:-5}" \
  -e NUM_WORKERS="${NUM_WORKERS:-8}" -e BUILD_WORKERS="${BUILD_WORKERS:-6}" -e TORCH_COMPILE="${TORCH_COMPILE:-1}" \
  -e CONTEXT_XY="${CONTEXT_XY:-256}" -e CONTEXT_Z="${CONTEXT_Z:-128}" \
  -e LAMBDA_V4_LINE_TEACHER="${LAMBDA_V4_LINE_TEACHER:-0.30}" -e V4_TEACHER_MIN_LINE_SCORE="${V4_TEACHER_MIN_LINE_SCORE:-0.30}" \
  -e LINE_CANDIDATE_THRESHOLD="${LINE_CANDIDATE_THRESHOLD:-0.08}" -e LINE_WEAK_THRESHOLD="${LINE_WEAK_THRESHOLD:-0.04}" \
  -e LINE_COMPETITION_RATIO="${LINE_COMPETITION_RATIO:-0.55}" -e LINE_MIN_VOXELS="${LINE_MIN_VOXELS:-3}" \
  --workdir /workspace/voxel_poleline "$IMAGE" bash -lc './run_v62_teacher_training.sh'

echo "Started $NAME"
echo "Follow: sudo docker logs -f $NAME"
