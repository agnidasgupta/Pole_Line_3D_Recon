#!/usr/bin/env bash
# Host-side launcher. No Python is run on the Nebius host.
set -euo pipefail
MODE=${V4_ALL_MODE:-all}
case "$MODE" in stage1|stage2|stage3|reconstruct|all) ;; *) echo "ERROR: invalid V4_ALL_MODE=$MODE" >&2; exit 2;; esac
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$HERE/v4_full_ops_common.sh"
v4_require_acceptance
v4_print_context

RAW=$(mktemp); TABLE_TMP=$(mktemp)
v4_build_session_inventory "$RAW"
[[ -s "$RAW" ]] || { echo "ERROR: no V4 session slices discovered below $V4_HOST_INPUT" >&2; exit 2; }
ALL_FILES=$(mktemp); MATCHED_FILES=$(mktemp); UNMATCHED=$(mktemp)
trap 'rm -f "$RAW" "$TABLE_TMP" "$ALL_FILES" "$MATCHED_FILES" "$UNMATCHED"' EXIT
find "$V4_HOST_INPUT" -type f \( -name '*.csv' -o -name '*.csv.gz' \) -print | sort > "$ALL_FILES"
cut -f3 "$RAW" | sort -u > "$MATCHED_FILES"
comm -23 "$ALL_FILES" "$MATCHED_FILES" > "$UNMATCHED"
if [[ -s "$UNMATCHED" ]]; then
  echo "ERROR: CSV inputs exist outside the V4 geography/sessionN_sliceM slice contract; refusing to call the run 'all data'." >&2
  echo "Unmatched files:" >&2; cat "$UNMATCHED" >&2
  exit 3
fi
{
  printf 'group_id\tslice_count\tmin_seq\tmax_seq\tmissing_sequence_count\tduplicate_source_count\tstatus\n'
  while IFS= read -r gid; do
    [[ -n "$gid" ]] || continue
    total=$(awk -F '\t' -v g="$gid" '$1==g{n++}END{print n+0}' "$RAW")
    seqs=$(awk -F '\t' -v g="$gid" '$1==g{print $2}' "$RAW" | sort -n -u)
    unique=$(printf '%s\n' "$seqs" | awk 'NF{n++}END{print n+0}')
    min=$(printf '%s\n' "$seqs" | awk 'NF{print;exit}')
    max=$(printf '%s\n' "$seqs" | awk 'NF{v=$1}END{print v}')
    dup=$((total-unique)); expected=$((max-min+1)); missing=$((expected-unique)); status=valid
    (( dup == 0 )) || status=duplicate_sources
    printf '%s\t%d\t%d\t%d\t%d\t%d\t%s\n' "$gid" "$unique" "$min" "$max" "$missing" "$dup" "$status"
  done < <(cut -f1 "$RAW" | sort -u)
} > "$TABLE_TMP"

if awk -F '\t' 'NR>1 && $6!=0{bad=1}END{exit !bad}' "$TABLE_TMP"; then
  echo "ERROR: duplicate source files exist for one or more slice sequences. Full-data run will not silently choose among duplicates." >&2
  cat "$TABLE_TMP" >&2
  exit 3
fi

INPUT_HASH=$(sha256sum "$RAW" | awk '{print $1}')
BASE="$V4_PROD_ROOT_HOST/full_dataset_runs/$V4_DEPLOY_SHORT"
mkdir -p "$BASE" "$V4_PROD_ROOT_HOST/full_dataset_packages"
LATEST_PTR="$V4_PROD_ROOT_HOST/LATEST_FULL_DATASET_RUN.txt"
RUN_ROOT_HOST=''

case "$MODE" in
  stage2|stage3|reconstruct)
    RUN_ROOT_HOST=$(cat "$LATEST_PTR" 2>/dev/null || true)
    [[ -n "$RUN_ROOT_HOST" && -d "$RUN_ROOT_HOST" ]] || { echo "ERROR: no prior full-dataset run found." >&2; exit 4; }
    case "$MODE" in
      stage2|reconstruct)
        [[ -f "$RUN_ROOT_HOST/PHASE1_STAGE1_OK.txt" ]] || { echo "ERROR: Stage 1 is not complete: $RUN_ROOT_HOST" >&2; exit 4; }
        ;;
      stage3)
        [[ -f "$RUN_ROOT_HOST/PHASE2_STAGE2_OK.txt" ]] || { echo "ERROR: Stage 2 is not complete: $RUN_ROOT_HOST" >&2; exit 4; }
        ;;
    esac
    old_deploy=$(awk -F= '$1=="V4_DEPLOY_FINGERPRINT"{print $2}' "$RUN_ROOT_HOST/run_context.env" 2>/dev/null || true)
    old_input=$(awk -F= '$1=="INPUT_INVENTORY_SHA256"{print $2}' "$RUN_ROOT_HOST/run_context.env" 2>/dev/null || true)
    [[ "$old_deploy" == "$V4_DEPLOY_FINGERPRINT" && "$old_input" == "$INPUT_HASH" ]] || { echo "ERROR: deployment or input inventory changed since upstream stage; refusing mixed-run execution." >&2; exit 5; }
    ;;
  stage1|all)
    prev=$(cat "$LATEST_PTR" 2>/dev/null || true)
    if [[ -n "$prev" && -d "$prev" && ! -f "$prev/COMPLETED.txt" ]]; then
      old_deploy=$(awk -F= '$1=="V4_DEPLOY_FINGERPRINT"{print $2}' "$prev/run_context.env" 2>/dev/null || true)
      old_input=$(awk -F= '$1=="INPUT_INVENTORY_SHA256"{print $2}' "$prev/run_context.env" 2>/dev/null || true)
      if [[ "$old_deploy" == "$V4_DEPLOY_FINGERPRINT" && "$old_input" == "$INPUT_HASH" ]]; then RUN_ROOT_HOST="$prev"; fi
    fi
    if [[ -z "$RUN_ROOT_HOST" ]]; then RUN_ROOT_HOST="$BASE/$(date -u +%Y%m%dT%H%M%SZ)"; fi
    ;;
esac

mkdir -p "$RUN_ROOT_HOST"
printf '%s\n' "$RUN_ROOT_HOST" > "$LATEST_PTR"
cp "$RAW" "$RUN_ROOT_HOST/raw_slice_inventory.tsv"
cp "$TABLE_TMP" "$RUN_ROOT_HOST/all_sessions.tsv"
cp "$TABLE_TMP" "$V4_PROD_ROOT_HOST/LATEST_ALL_SESSIONS.tsv"
OPS_BACKUP="$RUN_ROOT_HOST/v4_full_dataset_ops_$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
tar -C "$HERE" -czf "$OPS_BACKUP" .
cat > "$RUN_ROOT_HOST/run_context.env" <<CTX
CREATED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
V4_ACCEPTED_DIR=$V4_ACCEPTED_DIR
V4_HOST_INPUT=$V4_HOST_INPUT
V4_HOST_OUTPUTS=$V4_HOST_OUTPUTS
V4_DEPLOY_FINGERPRINT=$V4_DEPLOY_FINGERPRINT
V4_DEPLOY_SHORT=$V4_DEPLOY_SHORT
V4_RUNTIME_MODE=$V4_RUNTIME_MODE
V4_BATCH_SIZE=$V4_BATCH_SIZE
V4_EVALUATE_ALL_CORES=$V4_EVALUATE_ALL_CORES
V4_GPU_COORD_CHANNELS=$V4_GPU_COORD_CHANNELS
V4_FIXED_BATCH_SHAPE=$V4_FIXED_BATCH_SHAPE
INPUT_INVENTORY_SHA256=$INPUT_HASH
RUN_ROOT_HOST=$RUN_ROOT_HOST
CTX

RUN_ROOT=$(v4_to_container_output "$RUN_ROOT_HOST")
SESSION_TABLE=$(v4_to_container_output "$RUN_ROOT_HOST/all_sessions.tsv")
LOG="$RUN_ROOT/all_data_${MODE}.log"
RUN_ID=$(basename "$RUN_ROOT_HOST")
NAME="v4-all-${MODE}-${RUN_ID}-${V4_DEPLOY_SHORT}"
NAME=$(printf '%s' "$NAME" | tr -cd 'A-Za-z0-9_.-' | cut -c1-120)

if "${V4_DOCKER[@]}" inspect "$NAME" >/dev/null 2>&1; then
  running=$("${V4_DOCKER[@]}" inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null || printf false)
  if [[ "$running" == true ]]; then echo "Container already running: $NAME"; exit 0; fi
  "${V4_DOCKER[@]}" rm "$NAME" >/dev/null
fi
CID=$("${V4_DOCKER[@]}" run -d --name "$NAME" --gpus all \
  -e PYTHONPYCACHEPREFIX=/tmp/pycache -e MPLCONFIGDIR=/tmp/matplotlib -e XDG_CACHE_HOME=/tmp/xdg-cache \
  -e TORCH_HOME=/tmp/torch-home -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor -e TRITON_CACHE_DIR=/tmp/triton \
  -e CUDA_CACHE_PATH=/tmp/cuda-cache -e NUMBA_CACHE_DIR=/tmp/numba-cache -e JOBLIB_TEMP_FOLDER=/tmp/joblib -e TMPDIR=/tmp \
  -e MODE="$MODE" -e RUN_ROOT="$RUN_ROOT" -e SESSION_TABLE="$SESSION_TABLE" \
  -e MODEL_PATH="$V4_MODEL" -e CAL_PATH="$V4_CAL" -e STAGE2_BUNDLE="$V4_STAGE2" \
  -e V4_RUNTIME_MODE="$V4_RUNTIME_MODE" -e V4_BATCH_SIZE="$V4_BATCH_SIZE" \
  -e V4_EVALUATE_ALL_CORES="$V4_EVALUATE_ALL_CORES" -e V4_GPU_COORD_CHANNELS="$V4_GPU_COORD_CHANNELS" -e V4_FIXED_BATCH_SHAPE="$V4_FIXED_BATCH_SHAPE" \
  --mount "type=bind,source=$V4_ACCEPTED_DIR,target=/workspace/v4,readonly" \
  --mount "type=bind,source=$HERE,target=/workspace/v4_full_ops,readonly" \
  --mount "type=bind,source=$V4_HOST_OUTPUTS,target=/outputs" \
  --mount "type=bind,source=$V4_HOST_INPUT,target=/data/voxel_csv_combined,readonly" \
  --workdir /workspace/v4 "$V4_IMAGE" bash -lc "set -euo pipefail; bash /workspace/v4_full_ops/v4_all_data_inside_docker.sh 2>&1 | tee '$LOG'")
printf '%s\n' "$NAME" > "$RUN_ROOT_HOST/current_container.txt"
printf '%s\n' "$CID" > "$RUN_ROOT_HOST/current_container_id.txt"
echo "Started detached V4 all-data $MODE container: $NAME"
echo "Container ID: $CID"
echo "Run root: $RUN_ROOT_HOST"
echo "Sessions: $(awk 'NR>1{n++}END{print n+0}' "$TABLE_TMP")"
echo "Slices:   $(awk -F '\t' 'NR>1{s+=$2}END{print s+0}' "$TABLE_TMP")"
echo "Monitor:  $HERE/show_v4_all_data_state_on_nebius.sh"
