#!/usr/bin/env bash
set -euo pipefail
IMAGE="${IMAGE:-va-voxel-poleline:v6.2-three-stage}"
NAME="${NAME:-poleline-v62-teacher-reconstruction}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HOST_CODE="${HOST_CODE:-$SCRIPT_DIR}"
HOST_OUTPUTS="${HOST_OUTPUTS:-/workspace/voxel_poleline/outputs}"
HOST_RAW_INPUT_DIR="${HOST_RAW_INPUT_DIR:-/data/voxel_csv_combined}"

test -d "$HOST_CODE" || { echo "Code directory missing: $HOST_CODE"; exit 2; }
test -d "$HOST_OUTPUTS" || { echo "Outputs directory missing: $HOST_OUTPUTS"; exit 2; }
test -d "$HOST_RAW_INPUT_DIR" || { echo "Raw input directory missing: $HOST_RAW_INPUT_DIR"; exit 2; }

sudo docker rm -f "$NAME" >/dev/null 2>&1 || true
sudo docker run -d --name "$NAME" --shm-size=16g \
  --mount type=bind,source="$HOST_CODE",target=/workspace/voxel_poleline,readonly \
  --mount type=bind,source="$HOST_OUTPUTS",target=/outputs \
  --mount type=bind,source="$HOST_RAW_INPUT_DIR",target=/data/voxel_csv_combined,readonly \
  -e PYTHONPYCACHEPREFIX=/tmp/pycache \
  -e WORLD_UNITS_TO_FT="${WORLD_UNITS_TO_FT:-0.5}" -e MIN_POLE_SEPARATION_FT="${MIN_POLE_SEPARATION_FT:-10}" \
  -e MAX_SPAN_LENGTH_FT="${MAX_SPAN_LENGTH_FT:-450}" -e MAX_SPAN_SLICES="${MAX_SPAN_SLICES:-9}" \
  -e ALLOWED_POLE_HEIGHT_VARIATION_FT="${ALLOWED_POLE_HEIGHT_VARIATION_FT:-8}" -e MAX_POLE_HEIGHT_ADJUST_FT="${MAX_POLE_HEIGHT_ADJUST_FT:-8}" \
  -e RESUME_STAGE3="${RESUME_STAGE3:-0}" \
  --workdir /workspace/voxel_poleline "$IMAGE" bash -lc './run_v62_teacher_reconstruction.sh'

echo "Started $NAME"
echo "Follow: sudo docker logs -f $NAME"
