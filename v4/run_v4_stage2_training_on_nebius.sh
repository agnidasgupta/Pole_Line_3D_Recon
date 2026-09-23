#!/usr/bin/env bash
# Future/offline Stage-2 retraining launcher. Host shell only; Python runs in Docker.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IMAGE=${IMAGE:-va-v4-realtime:torch241-cu121}
if docker info >/dev/null 2>&1; then DOCKER=(docker); else DOCKER=(sudo docker); fi

# Choose an output root programmatically by finding the existing V4 Stage-1 checkpoint/calibration/dataset.
BEST=''; BEST_SCORE=-1
for ROOT in "$(cd "$HERE/.." && pwd)/outputs" /workspace/voxel_poleline/outputs /outputs /data/outputs; do
  [[ -d "$ROOT" ]] || continue
  SCORE=0
  find "$ROOT" -type f -name precision_best.pt -size +0c -print -quit 2>/dev/null | grep -q . && SCORE=$((SCORE+4)) || true
  find "$ROOT" -type f -name calibration.json -size +0c -print -quit 2>/dev/null | grep -q . && SCORE=$((SCORE+2)) || true
  find "$ROOT" -type d -name dataset_hardneg_v4opt_uncompressed -print -quit 2>/dev/null | grep -q . && SCORE=$((SCORE+4)) || true
  if (( SCORE > BEST_SCORE )); then BEST=$ROOT; BEST_SCORE=$SCORE; fi
done
[[ -n "$BEST" ]] || { echo 'ERROR: could not discover V4 output root for Stage2 retraining' >&2; exit 2; }
HOST_OUTPUTS=$(cd "$BEST" && pwd)
find_newest_file() { find "$HOST_OUTPUTS" -type f -name "$1" -size +0c -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-; }
DATASET_HOST=$(find "$HOST_OUTPUTS" -type d -name dataset_hardneg_v4opt_uncompressed -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
MODEL_HOST=$(find_newest_file precision_best.pt)
CAL_HOST=$(find_newest_file calibration.json)
[[ -d "$DATASET_HOST" && -s "$MODEL_HOST" && -s "$CAL_HOST" ]] || { echo 'ERROR: Stage2 training inputs could not be auto-discovered' >&2; exit 2; }
to_container() { case "$1" in "$HOST_OUTPUTS"/*) printf '/outputs/%s\n' "${1#"$HOST_OUTPUTS"/}";; *) return 2;; esac; }
DATASET=$(to_container "$DATASET_HOST"); MODEL=$(to_container "$MODEL_HOST"); CAL=$(to_container "$CAL_HOST")
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT_HOST="$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_retraining/$STAMP"
mkdir -p "$OUT_HOST"
OUT=$(to_container "$OUT_HOST")
NAME=${NAME:-v4-stage2-training-$STAMP}
"${DOCKER[@]}" rm -f "$NAME" >/dev/null 2>&1 || true
ID=$("${DOCKER[@]}" run -d --name "$NAME" --gpus all \
  -e PYTHONPYCACHEPREFIX=/tmp/pycache -e MPLCONFIGDIR=/tmp/matplotlib -e XDG_CACHE_HOME=/tmp/xdg-cache \
  -e TORCH_HOME=/tmp/torch-home -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor -e TRITON_CACHE_DIR=/tmp/triton \
  -e CUDA_CACHE_PATH=/tmp/cuda-cache -e NUMBA_CACHE_DIR=/tmp/numba-cache -e JOBLIB_TEMP_FOLDER=/tmp/joblib -e TMPDIR=/tmp \
  --mount "type=bind,source=$HERE,target=/workspace/v4,readonly" \
  --mount "type=bind,source=$HOST_OUTPUTS,target=/outputs" \
  --workdir /workspace/v4 \
  -e DATASET_DIR="$DATASET" -e OUT="$OUT" -e MODEL="$MODEL" -e CAL="$CAL" \
  -e BATCH_SIZE="${BATCH_SIZE:-12}" -e AMP="${AMP:-bf16}" -e COMPILE_MODEL="${COMPILE_MODEL:-0}" \
  -e EVALUATE_ALL_CORES="${EVALUATE_ALL_CORES:-1}" -e GPU_COORD_CHANNELS="${GPU_COORD_CHANNELS:-0}" -e RESUME="${RESUME:-1}" \
  "$IMAGE" bash ./run_v4_stage2_training.sh)
printf '%s\n' "$NAME" > "$OUT_HOST/container_name.txt"
printf '%s\n' "$ID" > "$OUT_HOST/container_id.txt"
printf 'output_root=%s\ndataset=%s\nmodel=%s\ncalibration=%s\n' "$OUT_HOST" "$DATASET_HOST" "$MODEL_HOST" "$CAL_HOST" > "$OUT_HOST/run_context.env"
echo "Started detached Stage2 training container: $NAME ($ID)"
echo "Persistent output: $OUT_HOST"
