#!/usr/bin/env bash
# Runs only inside the V4 Docker image. Never invoke this directly on the Nebius host.
set -euo pipefail
: "${MODE:?MODE required}"
: "${RUN_ROOT:?RUN_ROOT required}"
: "${SESSION_TABLE:?SESSION_TABLE required}"
: "${MODEL_PATH:?MODEL_PATH required}"
: "${CAL_PATH:?CAL_PATH required}"
: "${STAGE2_BUNDLE:?STAGE2_BUNDLE required}"
: "${V4_RUNTIME_MODE:?V4_RUNTIME_MODE required}"
: "${V4_BATCH_SIZE:?V4_BATCH_SIZE required}"
: "${V4_EVALUATE_ALL_CORES:?required}"
: "${V4_GPU_COORD_CHANNELS:?required}"
: "${V4_FIXED_BATCH_SHAPE:?required}"

case "$MODE" in
  stage1|stage2|stage3|reconstruct|all) ;;
  *) echo "ERROR: unsupported MODE=$MODE" >&2; exit 2 ;;
esac

export PYTHONPATH=/workspace/v4
mkdir -p "$RUN_ROOT" "$RUN_ROOT/logs/stage1" "$RUN_ROOT/logs/stage1_export" "$RUN_ROOT/logs/stage2" "$RUN_ROOT/logs/stage3" "$RUN_ROOT/timings/stage2" "$RUN_ROOT/timings/stage3" "$RUN_ROOT/metrics"
FAIL_MARKER="$RUN_ROOT/FAILED.txt"
rm -f "$FAIL_MARKER"
SUCCESS=0
on_exit() {
  rc=$?
  if [[ $SUCCESS -ne 1 ]]; then
    {
      echo "FAILED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      echo "EXIT_CODE=$rc"
      echo "MODE=$MODE"
      echo "LAST_COMMAND=${BASH_COMMAND:-unknown}"
    } > "$FAIL_MARKER"
  fi
  exit "$rc"
}
trap on_exit EXIT

safe_id() { printf '%s' "$1" | sed -E 's/[^A-Za-z0-9_.-]+/__/g'; }
now_ns() { date +%s%N; }
record_wall() {
  local phase=$1 gid=$2 start=$3 end=$4
  local ms=$(( (end-start)/1000000 ))
  printf '%s\t%s\t%d\n' "$phase" "$gid" "$ms" >> "$RUN_ROOT/phase_wall_times.tsv"
}
if [[ ! -f "$RUN_ROOT/phase_wall_times.tsv" ]]; then
  printf 'phase\tgroup_id\twall_ms\n' > "$RUN_ROOT/phase_wall_times.tsv"
fi

run_stage1() {
  echo "===== PHASE 1: STAGE 1 INFERENCE FOR ALL SESSIONS ====="
  while IFS=$'\t' read -r gid slice_count min_seq max_seq missing duplicate status; do
    [[ "$gid" == "group_id" ]] && continue
    [[ "$status" == "valid" ]] || { echo "ERROR: invalid session in execution table: $gid status=$status" >&2; exit 2; }
    sid=$(safe_id "$gid")
    s1="$RUN_ROOT/stage1/$sid"; exp="$RUN_ROOT/stage1_inference/$sid"
    mkdir -p "$s1" "$exp"
    start=$(now_ns)
    python /workspace/v4/run_v4_stage1.py \
      --input_dir /data/voxel_csv_combined \
      --session_filter "$gid" \
      --output_dir "$s1" \
      --model_path "$MODEL_PATH" \
      --calibration_json "$CAL_PATH" \
      --batch_size "$V4_BATCH_SIZE" \
      --amp "${AMP:-bf16}" \
      --compile_model "${COMPILE_MODEL:-0}" \
      --evaluate_all_cores "$V4_EVALUATE_ALL_CORES" \
      --gpu_coord_channels "$V4_GPU_COORD_CHANNELS" \
      --fixed_batch_shape "$V4_FIXED_BATCH_SHAPE" \
      --resume 1 2>&1 | tee "$RUN_ROOT/logs/stage1/$sid.log"
    end=$(now_ns); record_wall stage1 "$gid" "$start" "$end"

    start=$(now_ns)
    python /workspace/v4_full_ops/export_score_v4_stage1.py \
      --stage1_dir "$s1" --output_dir "$exp" --session_filter "$gid" \
      --calibration_json "$CAL_PATH" --resume 1 2>&1 | tee "$RUN_ROOT/logs/stage1_export/$sid.log"
    end=$(now_ns); record_wall stage1_export "$gid" "$start" "$end"
  done < "$SESSION_TABLE"
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_ROOT/PHASE1_STAGE1_OK.txt"
  python /workspace/v4_full_ops/summarize_v4_all_data.py --run_root "$RUN_ROOT" --session_table "$SESSION_TABLE"
}

run_stage2() {
  [[ -f "$RUN_ROOT/PHASE1_STAGE1_OK.txt" ]] || { echo "ERROR: Stage 1 phase is not complete: $RUN_ROOT/PHASE1_STAGE1_OK.txt" >&2; exit 3; }
  echo "===== PHASE 2: STAGE 2 PARAMETRIC RECONSTRUCTION FOR ALL SESSIONS ====="
  while IFS=$'\t' read -r gid slice_count min_seq max_seq missing duplicate status; do
    [[ "$gid" == "group_id" ]] && continue
    [[ "$status" == "valid" ]] || continue
    sid=$(safe_id "$gid")
    s1="$RUN_ROOT/stage1/$sid"; s2="$RUN_ROOT/stage2/$sid"; t2="$RUN_ROOT/timings/stage2/$sid.csv"
    mkdir -p "$s2"
    start=$(now_ns)
    python /workspace/v4_full_ops/profile_stage2_session.py \
      --stage1_dir "$s1" --output_dir "$s2" --session_filter "$gid" \
      --stage2_bundle "$STAGE2_BUNDLE" --timing_csv "$t2" --resume 1 2>&1 | tee "$RUN_ROOT/logs/stage2/$sid.log"
    end=$(now_ns); record_wall stage2 "$gid" "$start" "$end"
  done < "$SESSION_TABLE"
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_ROOT/PHASE2_STAGE2_OK.txt"
  python /workspace/v4_full_ops/summarize_v4_all_data.py --run_root "$RUN_ROOT" --session_table "$SESSION_TABLE"
}

run_stage3() {
  [[ -f "$RUN_ROOT/PHASE2_STAGE2_OK.txt" ]] || { echo "ERROR: Stage 2 phase is not complete: $RUN_ROOT/PHASE2_STAGE2_OK.txt" >&2; exit 4; }
  echo "===== PHASE 3: ROLLING 450-FT STAGE 3 RECONSTRUCTION FOR ALL SESSIONS ====="
  while IFS=$'\t' read -r gid slice_count min_seq max_seq missing duplicate status; do
    [[ "$gid" == "group_id" ]] && continue
    [[ "$status" == "valid" ]] || continue
    sid=$(safe_id "$gid")
    s2="$RUN_ROOT/stage2/$sid"; s3="$RUN_ROOT/stage3/$sid"; t3="$RUN_ROOT/timings/stage3/$sid.csv"
    mkdir -p "$s3"
    start=$(now_ns)
    python /workspace/v4_full_ops/profile_stage3_session.py \
      --stage2_dir "$s2" --output_dir "$s3" --session_filter "$gid" --timing_csv "$t3" \
      --max_sequence_gap 9 --slice_length_ft 50 --max_span_length_ft 450 --resume 1 2>&1 | tee "$RUN_ROOT/logs/stage3/$sid.log"
    end=$(now_ns); record_wall stage3 "$gid" "$start" "$end"
  done < "$SESSION_TABLE"
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_ROOT/PHASE3_STAGE3_OK.txt"

  echo "===== FINAL AGGREGATION ====="
  python /workspace/v4_full_ops/summarize_v4_all_data.py --run_root "$RUN_ROOT" --session_table "$SESSION_TABLE"
  cat > "$RUN_ROOT/COMPLETED.txt" <<DONE
completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
runtime_mode=$V4_RUNTIME_MODE
batch_size=$V4_BATCH_SIZE
max_sequence_gap=9
slice_length_ft=50
max_span_length_ft=450
DONE
}

case "$MODE" in
  stage1) run_stage1 ;;
  stage2) run_stage2 ;;
  stage3) run_stage3 ;;
  reconstruct) run_stage2; run_stage3 ;;
  all) run_stage1; run_stage2; run_stage3 ;;
esac

SUCCESS=1
trap - EXIT
rm -f "$FAIL_MARKER"
echo "V4_ALL_AVAILABLE_DATA_${MODE^^}_OK"
