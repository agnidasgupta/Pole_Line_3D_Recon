#!/usr/bin/env bash
set -euo pipefail

SESSION_FILTER="${SESSION_FILTER:?set SESSION_FILTER, e.g. 59768101-C4990BB-2026/session3}"
IMAGE="${IMAGE:-va-voxel-poleline:v6.2-three-stage}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HOST_CODE="${HOST_CODE:-$SCRIPT_DIR}"
HOST_OUTPUTS="${HOST_OUTPUTS:-/workspace/voxel_poleline/outputs}"
HOST_RAW_INPUT_DIR="${HOST_RAW_INPUT_DIR:-/data/voxel_csv_combined}"
V62_CONTAINER="${V62_CONTAINER:-/outputs/poleline_voxel_run_session_groups/v62_teacher_recall}"
SAFE_SESSION="$(printf '%s' "$SESSION_FILTER" | sed 's#[^A-Za-z0-9_.-]#_#g')"
TIMING_CONTAINER="${TIMING_CONTAINER:-$V62_CONTAINER/session_timing/$SAFE_SESSION}"
NAME="${NAME:-poleline-v62-session-infer-timed}"

test -d "$HOST_CODE" || { echo "Missing code dir: $HOST_CODE"; exit 2; }
test -d "$HOST_OUTPUTS" || { echo "Missing outputs dir: $HOST_OUTPUTS"; exit 2; }
test -d "$HOST_RAW_INPUT_DIR" || { echo "Missing raw input dir: $HOST_RAW_INPUT_DIR"; exit 2; }

sudo docker rm -f "$NAME" >/dev/null 2>&1 || true
sudo docker run --rm --name "$NAME" --gpus all --shm-size=64g \
  --mount type=bind,source="$HOST_CODE",target=/workspace/voxel_poleline,readonly \
  --mount type=bind,source="$HOST_OUTPUTS",target=/outputs \
  --mount type=bind,source="$HOST_RAW_INPUT_DIR",target=/data/voxel_csv_combined,readonly \
  -e PYTHONPYCACHEPREFIX=/tmp/pycache \
  --workdir /workspace/voxel_poleline \
  "$IMAGE" \
  python time_v62_one_session_inference.py \
    --session "$SESSION_FILTER" \
    --input_dir /data/voxel_csv_combined \
    --output_dir "$TIMING_CONTAINER/inference" \
    --code_dir /workspace/voxel_poleline \
    --model_path "$V62_CONTAINER/selected/v6_stage1_selected.pt" \
    --calibration_json "$V62_CONTAINER/selected/calibration.json" \
    --local_refiner_bundle "$V62_CONTAINER/stage2_refiner/local_refiner_bundle.joblib" \
    --batch_size "${INFER_BATCH:-5}" \
    --build_workers "${BUILD_WORKERS:-6}" \
    --compile_model "${TORCH_COMPILE:-1}" \
    --resume "${RESUME:-0}"

echo ""
echo "Inference timing output (host):"
echo "$HOST_OUTPUTS/poleline_voxel_run_session_groups/v62_teacher_recall/session_timing/$SAFE_SESSION/inference"
