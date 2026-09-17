#!/usr/bin/env bash
set -uo pipefail

TOOL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP_REPO=${EXP_REPO:-$(cd "$TOOL_DIR/../.." && pwd)}
FULL_OPS="$EXP_REPO/ops/v4_full_dataset"
HOST_OUTPUTS=${HOST_OUTPUTS:-/workspace/voxel_poleline/outputs}
HOST_INPUT=${HOST_INPUT:-/data/voxel_csv_combined}
IMAGE=${IMAGE:-va-v4-realtime:torch241-cu121}
BASELINE_RUN=${BASELINE_RUN_HOST:-$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_production/full_dataset_runs/d9977c39c443f5fa14f8/20260825T203403Z}
MODEL=${MODEL_HOST:-$HOST_OUTPUTS/poleline_voxel_run_session_groups/precision_v4/train/precision_best.pt}
CALIBRATION=${CALIBRATION_HOST:-$HOST_OUTPUTS/poleline_voxel_run_session_groups/precision_v4/full_val/calibration.json}
RUN_STAMP=${RUN_STAMP:?RUN_STAMP is required}
VARIANT_NAME=${VARIANT_NAME:?VARIANT_NAME is required}
RESUME=${RESUME:-1}
EXPECTED_SESSIONS=${EXPECTED_SESSIONS:-30}
SESSION_TIMEOUT_SECONDS=${SESSION_TIMEOUT_SECONDS:-7200}
SCORE_ATOL=${SCORE_ATOL:-0}
ONLY_GROUP_ID=${ONLY_GROUP_ID:-}
COMPILE_MODEL=${COMPILE_MODEL:-0}
COMPILE_MODE=${COMPILE_MODE:-default}
BATCH_SIZE=${BATCH_SIZE:-12}
CHANNELS_LAST=${CHANNELS_LAST:-1}
PINNED_D2H=${PINNED_D2H:-0}
DETAILED_CUDA_TIMING=${DETAILED_CUDA_TIMING:-1}
RETAIN_GATHER_HOST_BUFFERS=${RETAIN_GATHER_HOST_BUFFERS:-0}
PRUNE_EMBEDDING_HEAD=${PRUNE_EMBEDDING_HEAD:-0}
WARMUP_ITERATIONS=${WARMUP_ITERATIONS:-0}
RUN_ID="${RUN_STAMP}_${VARIANT_NAME}"
RUN_ROOT="$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_production/stage1_opt2_experiments/$RUN_ID"
RUN_C="/outputs/poleline_voxel_run_session_groups/v4_production/stage1_opt2_experiments/$RUN_ID"
DRIVER_LOG="$RUN_ROOT/STAGE1_OPT2_DRIVER.log"

fail() { echo "ERROR: $*" >&2; exit 1; }
host_to_container_output() {
  case "$1" in
    "$HOST_OUTPUTS") printf '/outputs\n' ;;
    "$HOST_OUTPUTS"/*) printf '/outputs%s\n' "${1#$HOST_OUTPUTS}" ;;
    *) fail "path is outside output mount: $1" ;;
  esac
}
context_value() {
  local key=$1
  awk -F= -v key="$key" '$1==key {sub(/^[^=]*=/, ""); print; exit}' "$BASELINE_RUN/run_context.env"
}
group_id() {
  awk -F',' 'NR==1 {for(i=1;i<=NF;i++){gsub(/^"|"$/, "", $i); if($i=="group_id")g=i} next}
    NR>1 && g {v=$g; gsub(/^"|"$/, "", v); gsub(/\r/, "", v); if(v!=""){print v; exit}}' "$1"
}

[[ "$RUN_STAMP" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || fail "RUN_STAMP must be UTC YYYYMMDDTHHMMSSZ"
[[ "$VARIANT_NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || fail "VARIANT_NAME may contain only letters, digits, dot, underscore, and hyphen"
[[ "$RESUME" =~ ^[01]$ ]] || fail "RESUME must be 0 or 1"
[[ "$EXPECTED_SESSIONS" =~ ^[0-9]+$ ]] || fail "EXPECTED_SESSIONS must be an integer"
[[ "$SESSION_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] || fail "SESSION_TIMEOUT_SECONDS must be an integer"
[[ "$COMPILE_MODEL" = 0 ]] || fail "production-preserving experiments require COMPILE_MODEL=0"
[[ "$CHANNELS_LAST" = 1 ]] || fail "production-preserving experiments require CHANNELS_LAST=1"
[[ "$PINNED_D2H" =~ ^[01]$ ]] || fail "PINNED_D2H must be 0 or 1"
[[ "$DETAILED_CUDA_TIMING" = 1 ]] || fail "accepted E0/E3 experiments require DETAILED_CUDA_TIMING=1"
[[ "$RETAIN_GATHER_HOST_BUFFERS" =~ ^[01]$ ]] || fail "RETAIN_GATHER_HOST_BUFFERS must be 0 or 1"
[[ "$PRUNE_EMBEDDING_HEAD" = 0 ]] || fail "production-preserving experiments require PRUNE_EMBEDDING_HEAD=0"
[[ "$WARMUP_ITERATIONS" = 0 ]] || fail "production-preserving experiments require WARMUP_ITERATIONS=0"
[[ "$BATCH_SIZE" = 12 ]] || fail "production-preserving experiments require BATCH_SIZE=12"
[[ "$COMPILE_MODE" = default ]] || fail "production-preserving experiments require COMPILE_MODE=default"
for path in "$EXP_REPO/.git" "$TOOL_DIR" "$FULL_OPS/export_score_v4_stage1.py" "$HOST_INPUT" "$BASELINE_RUN/PHASE1_STAGE1_OK.txt" "$BASELINE_RUN/run_context.env" "$MODEL" "$CALIBRATION"; do
  [ -e "$path" ] || fail "missing required path: $path"
done
for name in v4_realtime_core_opt2.py run_v4_stage1_opt2.py self_test_v4_stage1_opt2.py compare_v4_stage1_opt2_quality.py summarize_v4_stage1_opt2_timing.py; do
  [ -f "$TOOL_DIR/$name" ] || fail "missing $TOOL_DIR/$name"
done
[ "$RUN_ROOT" != "$BASELINE_RUN" ] || fail "candidate run must not overwrite baseline"

[ "$(context_value V4_RUNTIME_MODE)" = active_gpu ] || fail "baseline runtime was not active_gpu"
[ "$(context_value V4_BATCH_SIZE)" = 12 ] || fail "baseline batch size was not 12"
[ "$(context_value V4_EVALUATE_ALL_CORES)" = 0 ] || fail "baseline did not use active cores"
[ "$(context_value V4_GPU_COORD_CHANNELS)" = 1 ] || fail "baseline did not use GPU coordinate channels"
[ "$(context_value V4_FIXED_BATCH_SHAPE)" = 1 ] || fail "baseline did not use fixed batch shape"
current_model_sha=$(sha256sum "$MODEL" | awk '{print $1}')
current_cal_sha=$(sha256sum "$CALIBRATION" | awk '{print $1}')
MODEL_C=$(host_to_container_output "$MODEL")
CALIBRATION_C=$(host_to_container_output "$CALIBRATION")
BASELINE_META=$(find "$BASELINE_RUN/stage1" -type f -name '*_stage1.json' -print -quit)
[ -n "$BASELINE_META" ] && [ -s "$BASELINE_META" ] || fail "baseline Stage1 metadata is missing"
baseline_model_path=$(sed -n 's/^[[:space:]]*"model_path": "\(.*\)",*$/\1/p' "$BASELINE_META" | head -n 1)
baseline_calibration_path=$(sed -n 's/^[[:space:]]*"calibration_json": "\(.*\)",*$/\1/p' "$BASELINE_META" | head -n 1)
[ "$baseline_model_path" = "$MODEL_C" ] || {
  echo "baseline_stage1_model_path=$baseline_model_path" >&2
  echo "candidate_stage1_model_path=$MODEL_C" >&2
  fail "Stage1 model path differs from accepted baseline metadata"
}
[ "$baseline_calibration_path" = "$CALIBRATION_C" ] || {
  echo "baseline_stage1_calibration_path=$baseline_calibration_path" >&2
  echo "candidate_stage1_calibration_path=$CALIBRATION_C" >&2
  fail "Stage1 calibration path differs from accepted baseline metadata"
}

mkdir -p "$RUN_ROOT"/{stage1,stage1_inference,logs/stage1,logs/stage1_export,timings/stage1,status,metrics}
printf '%s\n' "$RUN_ROOT" > /home/agni/LATEST_V4_STAGE1_OPT2_RUN.txt
printf '%s\n' "$DRIVER_LOG" > /home/agni/LATEST_V4_STAGE1_OPT2_DRIVER_LOG.txt
printf '%s\n' "$BASELINE_RUN" > /home/agni/LATEST_V4_STAGE1_OPT2_BASELINE.txt

exec > >(tee -a "$DRIVER_LOG") 2>&1
echo "============================================================"
echo "V4 STAGE1 PRODUCTION-EQUIVALENT PERFORMANCE EXPERIMENT"
echo "============================================================"
echo "run_stamp=$RUN_STAMP"
echo "variant_name=$VARIANT_NAME"
echo "run_root=$RUN_ROOT"
echo "baseline_run=$BASELINE_RUN"
echo "model_sha256=$current_model_sha"
echo "calibration_sha256=$current_cal_sha"
echo "baseline_stage1_model_path=$baseline_model_path"
echo "baseline_stage1_calibration_path=$baseline_calibration_path"
echo "runtime=active_gpu amp=bf16 batch_size=$BATCH_SIZE fixed_batch_shape=1"
echo "compile_model=$COMPILE_MODEL compile_mode=$COMPILE_MODE channels_last=$CHANNELS_LAST"
echo "pinned_d2h=$PINNED_D2H prune_embedding_head=$PRUNE_EMBEDDING_HEAD warmup_iterations=$WARMUP_ITERATIONS"
echo "detailed_cuda_timing=$DETAILED_CUDA_TIMING"
echo "retain_gather_host_buffers=$RETAIN_GATHER_HOST_BUFFERS"
echo "only_group_id=${ONLY_GROUP_ID:-ALL} expected_sessions=$EXPECTED_SESSIONS"
echo "score_atol=$SCORE_ATOL"

echo "===== COMPILE AND SELF-TEST IN CUDA DOCKER ====="
docker run --rm --gpus all \
  --mount "type=bind,source=$EXP_REPO/v4,target=/workspace/v4,readonly" \
  --mount "type=bind,source=$TOOL_DIR,target=/workspace/opt,readonly" \
  --workdir /workspace/v4 -e PYTHONPATH=/workspace/v4:/workspace/opt \
  -e PYTHONPYCACHEPREFIX=/tmp/pycache "$IMAGE" bash -lc '
    set -euo pipefail
    python -m py_compile /workspace/opt/*.py
    python /workspace/opt/self_test_v4_stage1_opt2.py
  ' || fail "Docker compile/self-test failed"

mapfile -t MANIFESTS < <(find "$BASELINE_RUN/stage1" -mindepth 2 -maxdepth 2 -type f -name stage1_manifest.csv | sort)
if [ -n "$ONLY_GROUP_ID" ]; then
  FILTERED_MANIFESTS=()
  for manifest in "${MANIFESTS[@]}"; do
    [ "$(group_id "$manifest")" = "$ONLY_GROUP_ID" ] && FILTERED_MANIFESTS+=("$manifest")
  done
  MANIFESTS=("${FILTERED_MANIFESTS[@]}")
fi
[ "${#MANIFESTS[@]}" -eq "$EXPECTED_SESSIONS" ] || fail "expected $EXPECTED_SESSIONS baseline sessions, found ${#MANIFESTS[@]}"

BASELINE_C=$(host_to_container_output "$BASELINE_RUN")
printf 'group_id\tsid\n' > "$RUN_ROOT/session_map.tsv"
cat > "$RUN_ROOT/RUN_INFO.txt" <<EOF
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
experiment=v4_stage1_active_gpu_production_equivalent_opt2
variant_name=$VARIANT_NAME
repository=$EXP_REPO
branch=$(git -C "$EXP_REPO" branch --show-current)
commit=$(git -C "$EXP_REPO" rev-parse HEAD)
baseline_run=$BASELINE_RUN
model_sha256=$current_model_sha
calibration_sha256=$current_cal_sha
baseline_stage1_model_path=$baseline_model_path
baseline_stage1_calibration_path=$baseline_calibration_path
stage2_dependency=false
runtime_mode=active_gpu
amp=bf16
batch_size=$BATCH_SIZE
compile_model=$COMPILE_MODEL
compile_mode=$COMPILE_MODE
channels_last=$CHANNELS_LAST
pinned_d2h=$PINNED_D2H
detailed_cuda_timing=$DETAILED_CUDA_TIMING
retain_gather_host_buffers=$RETAIN_GATHER_HOST_BUFFERS
prune_embedding_head=$PRUNE_EMBEDDING_HEAD
warmup_iterations=$WARMUP_ITERATIONS
only_group_id=$ONLY_GROUP_ID
comparison_purpose=implementation_equivalence_only;incomplete_labels_are_not_ground_truth
evaluate_all_cores=0
gpu_coord_channels=1
fixed_batch_shape=1
patch_size=64
core_size=48
output_layout=stage1/<SID>/stage1_scores and stage1_inference/<SID>/inference_rows
EOF

accepted=0
failed=0
index=0
for manifest in "${MANIFESTS[@]}"; do
  index=$((index+1))
  gid=$(group_id "$manifest")
  sid=$(basename "$(dirname "$manifest")")
  expected_sid=$(printf '%s' "$gid" | sed -E 's/[^A-Za-z0-9_.-]+/__/g')
  [ "$sid" = "$expected_sid" ] || fail "baseline sid mismatch for $gid: $sid != $expected_sid"
  printf '%s\t%s\n' "$gid" "$sid" >> "$RUN_ROOT/session_map.tsv"
  candidate_stage1="$RUN_ROOT/stage1/$sid"
  candidate_export="$RUN_ROOT/stage1_inference/$sid"
  candidate_stage1_c="$RUN_C/stage1/$sid"
  candidate_export_c="$RUN_C/stage1_inference/$sid"
  timing_c="$RUN_C/timings/stage1/$sid.csv"
  progress_c="$RUN_C/status/$sid.progress.json"
  report_c="$RUN_C/status/$sid.production_equivalence.json"
  log="$RUN_ROOT/logs/stage1/$sid.log"
  export_log="$RUN_ROOT/logs/stage1_export/$sid.log"
  ok="$RUN_ROOT/status/$sid.stage1.ok"
  bad="$RUN_ROOT/status/$sid.stage1.failed"
  report="$RUN_ROOT/status/$sid.production_equivalence.json"
  mkdir -p "$candidate_stage1" "$candidate_export"
  echo "============================================================"
  echo "SESSION $index/${#MANIFESTS[@]} gid=$gid sid=$sid"
  if [ "$RESUME" = 1 ] && [ -s "$ok" ] && [ -s "$report" ]; then
    echo "SESSION_REUSED"
    accepted=$((accepted+1))
    continue
  fi
  rm -f "$ok" "$bad" "$report"
  start=$(date +%s)
  set +e
  timeout --signal=TERM --kill-after=120 "$SESSION_TIMEOUT_SECONDS" docker run --rm --gpus all \
    --mount "type=bind,source=$EXP_REPO/v4,target=/workspace/v4,readonly" \
    --mount "type=bind,source=$TOOL_DIR,target=/workspace/opt,readonly" \
    --mount "type=bind,source=$HOST_OUTPUTS,target=/outputs" \
    --mount "type=bind,source=$HOST_INPUT,target=/data/voxel_csv_combined,readonly" \
    --workdir /workspace/v4 -e PYTHONPATH=/workspace/v4:/workspace/opt \
    -e PYTHONPYCACHEPREFIX=/tmp/pycache -e MPLCONFIGDIR=/tmp/matplotlib \
    -e XDG_CACHE_HOME=/tmp/xdg-cache "$IMAGE" \
    python /workspace/opt/run_v4_stage1_opt2.py \
      --input_dir /data/voxel_csv_combined --session_filter "$gid" \
      --output_dir "$candidate_stage1_c" --model_path "$MODEL_C" \
      --calibration_json "$CALIBRATION_C" --timing_csv "$timing_c" \
      --progress_json "$progress_c" --batch_size "$BATCH_SIZE" --amp bf16 \
      --compile_model "$COMPILE_MODEL" --compile_mode "$COMPILE_MODE" \
      --channels_last "$CHANNELS_LAST" --pinned_d2h "$PINNED_D2H" \
      --detailed_cuda_timing "$DETAILED_CUDA_TIMING" \
      --retain_gather_host_buffers "$RETAIN_GATHER_HOST_BUFFERS" \
      --prune_embedding_head "$PRUNE_EMBEDDING_HEAD" --warmup_iterations "$WARMUP_ITERATIONS" \
      --evaluate_all_cores 0 --gpu_coord_channels 1 \
      --fixed_batch_shape 1 --resume "$RESUME" --max_slices 0 \
      > "$log" 2>&1
  rc=$?
  set -e

  if [ "$rc" -eq 0 ]; then
    set +e
    timeout --signal=TERM --kill-after=60 1800 docker run --rm \
      --mount "type=bind,source=$EXP_REPO/v4,target=/workspace/v4,readonly" \
      --mount "type=bind,source=$FULL_OPS,target=/workspace/v4_full_ops,readonly" \
      --mount "type=bind,source=$HOST_OUTPUTS,target=/outputs" \
      --mount "type=bind,source=$HOST_INPUT,target=/data/voxel_csv_combined,readonly" \
      --workdir /workspace/v4 -e PYTHONPATH=/workspace/v4 \
      -e PYTHONPYCACHEPREFIX=/tmp/pycache "$IMAGE" \
      python /workspace/v4_full_ops/export_score_v4_stage1.py \
        --stage1_dir "$candidate_stage1_c" --output_dir "$candidate_export_c" \
        --session_filter "$gid" --calibration_json "$CALIBRATION_C" --resume "$RESUME" \
        > "$export_log" 2>&1
    rc=$?
    set -e
  fi

  if [ "$rc" -eq 0 ]; then
    set +e
    docker run --rm \
      --mount "type=bind,source=$EXP_REPO/v4,target=/workspace/v4,readonly" \
      --mount "type=bind,source=$TOOL_DIR,target=/workspace/opt,readonly" \
      --mount "type=bind,source=$HOST_OUTPUTS,target=/outputs" \
      --workdir /workspace/v4 -e PYTHONPATH=/workspace/v4:/workspace/opt \
      "$IMAGE" python /workspace/opt/compare_v4_stage1_opt2_quality.py \
        --baseline-stage1 "$BASELINE_C/stage1/$sid" \
        --candidate-stage1 "$candidate_stage1_c" \
        --baseline-export "$BASELINE_C/stage1_inference/$sid" \
        --candidate-export "$candidate_export_c" \
        --calibration-json "$CALIBRATION_C" --score-atol "$SCORE_ATOL" \
        --report "$report_c" >> "$log" 2>&1
    guard_rc=$?
    set -e
    if [ "$guard_rc" -ne 0 ]; then
      printf 'gid=%s\nsid=%s\nexit_code=%s\nreason=production_output_equivalence_failed\nlog=%s\nreport=%s\n' \
        "$gid" "$sid" "$guard_rc" "$log" "$report" > "$bad"
      echo "PRODUCTION_OUTPUT_EQUIVALENCE_FAILED gid=$gid sid=$sid report=$report log=$log"
      tail -n 100 "$log"
      exit "$guard_rc"
    fi
  fi

  elapsed=$(( $(date +%s) - start ))
  if [ "$rc" -eq 0 ]; then
    printf 'gid=%s\nsid=%s\nelapsed_seconds=%s\nproduction_output_equivalent=true\n' \
      "$gid" "$sid" "$elapsed" > "$ok"
    accepted=$((accepted+1))
    echo "SESSION_ACCEPTED elapsed_seconds=$elapsed"
  else
    printf 'gid=%s\nsid=%s\nexit_code=%s\nelapsed_seconds=%s\nlog=%s\n' \
      "$gid" "$sid" "$rc" "$elapsed" "$log" > "$bad"
    failed=$((failed+1))
    echo "SESSION_REJECTED exit_code=$rc elapsed_seconds=$elapsed log=$log"
    tail -n 80 "$log"
  fi
done

if [ "$failed" -ne 0 ] || [ "$accepted" -ne "$EXPECTED_SESSIONS" ]; then
  echo "V4_STAGE1_OPT2_SESSION_EXECUTION_FAILED accepted=$accepted failed=$failed expected=$EXPECTED_SESSIONS"
  echo "Timing summarization skipped because no complete, production-equivalent candidate set exists."
  exit 1
fi

docker run --rm \
  --mount "type=bind,source=$TOOL_DIR,target=/workspace/opt,readonly" \
  --mount "type=bind,source=$HOST_OUTPUTS,target=/outputs" \
  "$IMAGE" python /workspace/opt/summarize_v4_stage1_opt2_timing.py \
    --candidate-timing-dir "$RUN_C/timings/stage1" \
    --baseline-run "$BASELINE_C" \
    --output "$RUN_C/STAGE1_TIMING_SESSION_AVERAGES.txt" \
  || fail "timing summary failed"

guarded=$(find "$RUN_ROOT/status" -maxdepth 1 -type f -name '*.production_equivalence.json' | wc -l | tr -d ' ')
find "$RUN_ROOT" -type f ! -name FILE_INVENTORY.txt -printf '%P\t%s\n' | sort > "$RUN_ROOT/FILE_INVENTORY.txt"
echo "accepted=$accepted failed=$failed production_equivalent=$guarded expected=$EXPECTED_SESSIONS"
if [ "$accepted" -eq "$EXPECTED_SESSIONS" ] && [ "$failed" -eq 0 ] && [ "$guarded" -eq "$EXPECTED_SESSIONS" ]; then
  printf 'completed_utc=%s\naccepted=%s\nfailed=0\nproduction_equivalent_sessions=%s\ncomparison_purpose=implementation_equivalence_only\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$accepted" "$guarded" \
    > "$RUN_ROOT/PHASE1_STAGE1_OK.txt"
  cp "$RUN_ROOT/PHASE1_STAGE1_OK.txt" "$RUN_ROOT/STAGE1_OPT2_PRODUCTION_EQUIVALENT_COMPLETE.txt"
  find "$RUN_ROOT" -type f ! -name FILE_INVENTORY.txt -printf '%P\t%s\n' | sort > "$RUN_ROOT/FILE_INVENTORY.txt"
  echo "V4_STAGE1_OPT2_ALL_SESSIONS_PRODUCTION_EQUIVALENT_AND_OK"
  exit 0
fi
echo "V4_STAGE1_OPT2_INCOMPLETE"
exit 1
