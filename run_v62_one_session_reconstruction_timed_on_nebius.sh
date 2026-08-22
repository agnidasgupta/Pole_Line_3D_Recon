#!/usr/bin/env bash
set -euo pipefail

SESSION_FILTER="${SESSION_FILTER:?set SESSION_FILTER, same session used for timed inference}"
IMAGE="${IMAGE:-va-voxel-poleline:v6.2-three-stage}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HOST_CODE="${HOST_CODE:-$SCRIPT_DIR}"
HOST_OUTPUTS="${HOST_OUTPUTS:-/workspace/voxel_poleline/outputs}"
HOST_RAW_INPUT_DIR="${HOST_RAW_INPUT_DIR:-/data/voxel_csv_combined}"
V62_CONTAINER="${V62_CONTAINER:-/outputs/poleline_voxel_run_session_groups/v62_teacher_recall}"
SAFE_SESSION="$(printf '%s' "$SESSION_FILTER" | sed 's#[^A-Za-z0-9_.-]#_#g')"
TIMING_CONTAINER="${TIMING_CONTAINER:-$V62_CONTAINER/session_timing/$SAFE_SESSION}"
MODE="${MODE:-both}"
NAME="${NAME:-poleline-v62-session-recon-timed}"

test -d "$HOST_CODE" || { echo "Missing code dir: $HOST_CODE"; exit 2; }
test -d "$HOST_OUTPUTS" || { echo "Missing outputs dir: $HOST_OUTPUTS"; exit 2; }
test -d "$HOST_RAW_INPUT_DIR" || { echo "Missing raw input dir: $HOST_RAW_INPUT_DIR"; exit 2; }
HOST_INF="$HOST_OUTPUTS/poleline_voxel_run_session_groups/v62_teacher_recall/session_timing/$SAFE_SESSION/inference"
test -s "$HOST_INF/inference_manifest.csv" || { echo "Missing timed-session inference manifest: $HOST_INF/inference_manifest.csv"; exit 2; }

extra=()
if [[ "${KEEP_ROLLING_OUTPUTS:-0}" == "1" ]]; then extra+=(--keep_rolling_outputs); fi

sudo docker rm -f "$NAME" >/dev/null 2>&1 || true
sudo docker run --rm --name "$NAME" --shm-size=64g \
  --mount type=bind,source="$HOST_CODE",target=/workspace/voxel_poleline,readonly \
  --mount type=bind,source="$HOST_OUTPUTS",target=/outputs \
  --mount type=bind,source="$HOST_RAW_INPUT_DIR",target=/data/voxel_csv_combined,readonly \
  -e PYTHONPYCACHEPREFIX=/tmp/pycache \
  --workdir /workspace/voxel_poleline \
  "$IMAGE" \
  python time_v62_one_session_reconstruction.py \
    --session "$SESSION_FILTER" \
    --inference_dir "$TIMING_CONTAINER/inference" \
    --metadata_dir /data/voxel_csv_combined \
    --output_dir "$TIMING_CONTAINER/reconstruction_timing" \
    --code_dir /workspace/voxel_poleline \
    --mode "$MODE" \
    --max_span_slices "${MAX_SPAN_SLICES:-9}" \
    --world_units_to_ft "${WORLD_UNITS_TO_FT:-0.5}" \
    "${extra[@]}"

echo ""
echo "Reconstruction timing output (host):"
echo "$HOST_OUTPUTS/poleline_voxel_run_session_groups/v62_teacher_recall/session_timing/$SAFE_SESSION/reconstruction_timing"
