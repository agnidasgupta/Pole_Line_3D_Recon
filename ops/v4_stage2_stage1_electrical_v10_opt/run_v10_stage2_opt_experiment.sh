#!/usr/bin/env bash
set -uo pipefail

EXP_REPO=${EXP_REPO:-/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10}
TOOL_DIR="$EXP_REPO/ops/v4_stage2_stage1_electrical_v10_opt"
HOST_OUTPUTS=${HOST_OUTPUTS:-/workspace/voxel_poleline/outputs}
IMAGE=${IMAGE:-va-v4-realtime:torch241-cu121}
STAGE1_BASELINE=${BASELINE_RUN_HOST:-$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_production/full_dataset_runs/d9977c39c443f5fa14f8/20260825T203403Z}
STAGE1_ROOT=${STAGE1_ROOT_HOST:-$STAGE1_BASELINE/stage1}
QUALITY_BASELINE_ROOT=${QUALITY_BASELINE_ROOT:?QUALITY_BASELINE_ROOT is required}
BUNDLE=${STAGE2_BUNDLE_HOST:-$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_realtime/stage2_refiner/local_refiner_bundle.joblib}
CALIBRATION=${CALIBRATION_HOST:-$HOST_OUTPUTS/poleline_voxel_run_session_groups/precision_v4/full_val/calibration.json}
RUN_STAMP=${RUN_STAMP:?RUN_STAMP is required}
RESUME=${RESUME:-1}
EXPECTED_SESSIONS=${EXPECTED_SESSIONS:-30}
SESSION_TIMEOUT_SECONDS=${SESSION_TIMEOUT_SECONDS:-7200}
PROFILE_GID=VELASCO_CUT_CP/session1
PROFILE_SID=VELASCO_CUT_CP__session1
RUN_ROOT="$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_stage23_quality/fault_tolerant_v10_${RUN_STAMP}"
RUN_C="/outputs/poleline_voxel_run_session_groups/v4_stage23_quality/fault_tolerant_v10_${RUN_STAMP}"
DRIVER_LOG="$RUN_ROOT/FAULT_TOLERANT_DRIVER.log"

fail() { echo "ERROR: $*" >&2; exit 1; }
host_to_container() {
  case "$1" in
    "$HOST_OUTPUTS") printf '/outputs\n' ;;
    "$HOST_OUTPUTS"/*) printf '/outputs%s\n' "${1#$HOST_OUTPUTS}" ;;
    *) fail "path outside output mount: $1" ;;
  esac
}
group_id() {
  awk -F',' 'NR==1 {for(i=1;i<=NF;i++){gsub(/^"|"$/, "", $i); if($i=="group_id")g=i} next}
    NR>1 && g {v=$g; gsub(/^"|"$/, "", v); gsub(/\r/, "", v); if(v!=""){print v; exit}}' "$1"
}

[[ "$RUN_STAMP" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || fail "bad RUN_STAMP"
[[ "$RESUME" =~ ^[01]$ ]] || fail "RESUME must be 0 or 1"
[[ "$EXPECTED_SESSIONS" =~ ^[0-9]+$ ]] || fail "bad EXPECTED_SESSIONS"
[[ "$SESSION_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] || fail "bad SESSION_TIMEOUT_SECONDS"
[ "$QUALITY_BASELINE_ROOT" != "$RUN_ROOT" ] || fail "optimized run root must differ from quality baseline"
for path in "$EXP_REPO/.git" "$STAGE1_ROOT" "$BUNDLE" "$CALIBRATION" "$QUALITY_BASELINE_ROOT/STAGE2_ONLY_COMPLETE.txt"; do [ -e "$path" ] || fail "missing $path"; done
for name in v4_stage2_stage1_electrical_tracks_opt.py run_v4_stage2_stage1_electrical_tracks_opt.py learn_velasco_stage1_electrical_profile_opt.py self_test_stage1_electrical_tracks_opt.py self_test_compare_v10_stage2_quality.py validate_v10_voxel_supported_stage2_opt.py compare_v10_stage2_quality.py summarize_v10_opt_timing.py; do
  [ -f "$TOOL_DIR/$name" ] || fail "missing $TOOL_DIR/$name"
done

mkdir -p "$RUN_ROOT"/{stage2,stage3,selection,logs/stage2,logs/stage3,timing/stage2,status}
printf '%s\n' "$RUN_ROOT" > /home/agni/LATEST_V10_STAGE2_OPT_RUN.txt
printf '%s\n' "$DRIVER_LOG" > /home/agni/LATEST_V10_STAGE2_OPT_DRIVER_LOG.txt
printf '%s\n' "$QUALITY_BASELINE_ROOT" > /home/agni/LATEST_V10_STAGE2_OPT_BASELINE.txt

exec > >(tee -a "$DRIVER_LOG") 2>&1
echo "============================================================"
echo "V10 STRICT VOXEL-SUPPORTED STAGE 2 PERFORMANCE EXPERIMENT"
echo "============================================================"
echo "run_stamp=$RUN_STAMP"
echo "run_root=$RUN_ROOT"
echo "quality_baseline=$QUALITY_BASELINE_ROOT"
echo "resume=$RESUME"
echo "session_timeout_seconds=$SESSION_TIMEOUT_SECONDS"

echo "===== COMPILE AND SELF-TEST IN DOCKER ====="
docker run --rm \
  --mount "type=bind,source=$EXP_REPO/v4,target=/workspace/v4,readonly" \
  --mount "type=bind,source=$TOOL_DIR,target=/workspace/quality,readonly" \
  --workdir /workspace/v4 -e PYTHONPATH=/workspace/v4:/workspace/quality \
  -e PYTHONPYCACHEPREFIX=/tmp/pycache "$IMAGE" bash -lc '
    set -euo pipefail
    python -m py_compile \
      /workspace/quality/v4_stage2_stage1_electrical_tracks_opt.py \
      /workspace/quality/run_v4_stage2_stage1_electrical_tracks_opt.py \
      /workspace/quality/learn_velasco_stage1_electrical_profile_opt.py \
      /workspace/quality/self_test_stage1_electrical_tracks_opt.py \
      /workspace/quality/self_test_compare_v10_stage2_quality.py \
      /workspace/quality/validate_v10_voxel_supported_stage2_opt.py \
      /workspace/quality/compare_v10_stage2_quality.py \
      /workspace/quality/summarize_v10_opt_timing.py
    python /workspace/quality/self_test_stage1_electrical_tracks_opt.py
    python /workspace/quality/self_test_compare_v10_stage2_quality.py
  ' || fail "Docker compile/self-test failed"

mapfile -t MANIFESTS < <(find "$STAGE1_ROOT" -mindepth 2 -maxdepth 2 -type f -name stage1_manifest.csv | sort)
[ "${#MANIFESTS[@]}" -eq "$EXPECTED_SESSIONS" ] || fail "expected $EXPECTED_SESSIONS sessions, found ${#MANIFESTS[@]}"
PROFILE_STAGE1="$STAGE1_ROOT/$PROFILE_SID"
PROFILE_STAGE1_C=$(host_to_container "$PROFILE_STAGE1")
CALIBRATION_C=$(host_to_container "$CALIBRATION")
BUNDLE_C=$(host_to_container "$BUNDLE")

echo "===== LEARN FIXED PROFILE FROM $PROFILE_GID ====="
docker run --rm \
  --mount "type=bind,source=$EXP_REPO/v4,target=/workspace/v4,readonly" \
  --mount "type=bind,source=$TOOL_DIR,target=/workspace/quality,readonly" \
  --mount "type=bind,source=$HOST_OUTPUTS,target=/outputs" \
  --workdir /workspace/v4 -e PYTHONPATH=/workspace/v4:/workspace/quality \
  "$IMAGE" python /workspace/quality/learn_velasco_stage1_electrical_profile_opt.py \
  --stage1_dir "$PROFILE_STAGE1_C" --calibration_json "$CALIBRATION_C" \
  --session_filter "$PROFILE_GID" --reference_min 0 --reference_max 19 \
  --fragment_min 20 --fragment_max 39 --output_dir "$RUN_C/selection" \
  || fail "profile learning failed"

PROFILE_C="$RUN_C/selection/selected_electrical_profile.json"
cat > "$RUN_ROOT/RUN_INFO.txt" <<EOF
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
layout_contract=v4_stage23_quality/fault_tolerant_v10_<UTC>/stage2/<SID>
experiment=v10_stage2_behavior_preserving_performance_opt1_fix2
accepted_stage2_commit=ed852df
quality_baseline=$QUALITY_BASELINE_ROOT
repository=$EXP_REPO
branch=$(git -C "$EXP_REPO" branch --show-current)
commit=$(git -C "$EXP_REPO" rev-parse HEAD)
stage1_source=$STAGE1_ROOT
stage1_rerun=false
runtime_gt_usage=false
synthetic_line_voxels=0
disconnected_fragment_bridge_allowed=false
line_geometry_must_stay_inside_stage1_voxel_cells=true
pole_attachment_requires_stage1_voxel_contact=true
open_line_endpoints_preserved=true
stage3_ran=false
expected_sessions=$EXPECTED_SESSIONS
EOF

QUALITY_BASELINE_C=$(host_to_container "$QUALITY_BASELINE_ROOT")

: > "$RUN_ROOT/session_map.tsv"
accepted=0
failed=0
index=0
for manifest in "${MANIFESTS[@]}"; do
  index=$((index+1))
  gid=$(group_id "$manifest")
  sid=$(basename "$(dirname "$manifest")")
  expected_sid=$(printf '%s' "$gid" | sed -E 's/[^A-Za-z0-9_.-]+/__/g')
  [ "$sid" = "$expected_sid" ] || { echo "SESSION_REJECTED sid mismatch $sid $expected_sid"; failed=$((failed+1)); continue; }
  printf '%s\t%s\n' "$gid" "$sid" >> "$RUN_ROOT/session_map.tsv"
  ok="$RUN_ROOT/status/$sid.stage2.ok"
  bad="$RUN_ROOT/status/$sid.stage2.failed"
  report="$RUN_ROOT/status/$sid.voxel_support_validation.json"
  equivalence="$RUN_ROOT/status/$sid.quality_equivalence.json"
  session="$RUN_ROOT/stage2/$sid"
  session_c="$RUN_C/stage2/$sid"
  stage1_session=$(dirname "$manifest")
  stage1_c=$(host_to_container "$stage1_session")
  timing_c="$RUN_C/timing/stage2/$sid.csv"
  log="$RUN_ROOT/logs/stage2/$sid.log"
  mkdir -p "$session"
  echo "============================================================"
  echo "SESSION $index/${#MANIFESTS[@]} gid=$gid sid=$sid"
  if [ "$RESUME" = 1 ] && [ -s "$ok" ] && [ -s "$report" ] && [ -s "$equivalence" ]; then
    echo "SESSION_REUSED"
    accepted=$((accepted+1))
    continue
  fi
  rm -f "$ok" "$bad" "$report" "$equivalence"
  start=$(date +%s)
  set +e
  timeout --signal=TERM --kill-after=120 "$SESSION_TIMEOUT_SECONDS" docker run --rm \
    --mount "type=bind,source=$EXP_REPO/v4,target=/workspace/v4,readonly" \
    --mount "type=bind,source=$TOOL_DIR,target=/workspace/quality,readonly" \
    --mount "type=bind,source=$HOST_OUTPUTS,target=/outputs" \
    --workdir /workspace/v4 -e PYTHONPATH=/workspace/v4:/workspace/quality \
    -e PYTHONPYCACHEPREFIX=/tmp/pycache -e MPLCONFIGDIR=/tmp/matplotlib \
    "$IMAGE" python /workspace/quality/run_v4_stage2_stage1_electrical_tracks_opt.py \
    --stage1_dir "$stage1_c" --output_dir "$session_c" --session_filter "$gid" \
    --stage2_bundle "$BUNDLE_C" --calibration_json "$CALIBRATION_C" \
    --profile_json "$PROFILE_C" --timing_csv "$timing_c" \
    --resume "$RESUME" --max_slices 0 --write_voxel_audit 1 > "$log" 2>&1
  rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then
    set +e
    docker run --rm \
      --mount "type=bind,source=$TOOL_DIR,target=/workspace/quality,readonly" \
      --mount "type=bind,source=$HOST_OUTPUTS,target=/outputs" \
      "$IMAGE" python /workspace/quality/validate_v10_voxel_supported_stage2_opt.py \
      --session-dir "$session_c" --report "$RUN_C/status/$sid.voxel_support_validation.json" \
      >> "$log" 2>&1
    rc=$?
    set -e
  fi
  if [ "$rc" -eq 0 ]; then
    set +e
    docker run --rm \
      --mount "type=bind,source=$TOOL_DIR,target=/workspace/quality,readonly" \
      --mount "type=bind,source=$HOST_OUTPUTS,target=/outputs" \
      "$IMAGE" python /workspace/quality/compare_v10_stage2_quality.py \
      --baseline-session "$QUALITY_BASELINE_C/stage2/$sid" \
      --candidate-session "$session_c" \
      --report "$RUN_C/status/$sid.quality_equivalence.json" \
      >> "$log" 2>&1
    equivalence_rc=$?
    set -e
    if [ "$equivalence_rc" -ne 0 ]; then
      printf 'gid=%s\nsid=%s\nexit_code=%s\nreason=quality_equivalence_failed\nlog=%s\n' "$gid" "$sid" "$equivalence_rc" "$log" > "$bad"
      echo "QUALITY_EQUIVALENCE_FAILED gid=$gid sid=$sid log=$log"
      tail -n 80 "$log"
      exit "$equivalence_rc"
    fi
  fi
  elapsed=$(( $(date +%s) - start ))
  if [ "$rc" -eq 0 ]; then
    printf 'gid=%s\nsid=%s\nelapsed_seconds=%s\nvalidated=true\n' "$gid" "$sid" "$elapsed" > "$ok"
    accepted=$((accepted+1))
    echo "SESSION_ACCEPTED elapsed_seconds=$elapsed"
  else
    printf 'gid=%s\nsid=%s\nexit_code=%s\nelapsed_seconds=%s\nlog=%s\n' "$gid" "$sid" "$rc" "$elapsed" "$log" > "$bad"
    failed=$((failed+1))
    echo "SESSION_REJECTED exit_code=$rc elapsed_seconds=$elapsed log=$log"
    tail -n 40 "$log"
  fi
done

docker run --rm \
  --mount "type=bind,source=$TOOL_DIR,target=/workspace/quality,readonly" \
  --mount "type=bind,source=$HOST_OUTPUTS,target=/outputs" \
  "$IMAGE" python /workspace/quality/summarize_v10_opt_timing.py \
  --timing-dir "$RUN_C/timing/stage2" \
  --baseline-timing-dir "$QUALITY_BASELINE_C/timing/stage2" \
  --output "$RUN_C/STAGE2_TIMING_SESSION_AVERAGES.txt" \
  || fail "timing summary failed"

find "$RUN_ROOT" -type f -printf '%P\t%s\n' | sort > "$RUN_ROOT/FILE_INVENTORY.txt"
echo "accepted=$accepted failed=$failed expected=$EXPECTED_SESSIONS"
if [ "$accepted" -eq "$EXPECTED_SESSIONS" ] && [ "$failed" -eq 0 ]; then
  printf 'completed_utc=%s\naccepted=%s\nfailed=0\nstage3_ran=false\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$accepted" > "$RUN_ROOT/PHASE2_STAGE2_OK.txt"
  cp "$RUN_ROOT/PHASE2_STAGE2_OK.txt" "$RUN_ROOT/STAGE2_ONLY_COMPLETE.txt"
  find "$RUN_ROOT" -type f ! -name FILE_INVENTORY.txt -printf '%P\t%s\n' | sort > "$RUN_ROOT/FILE_INVENTORY.txt"
  echo "V10_STAGE2_OPT_ALL_SESSIONS_EQUIVALENT_AND_OK"
  exit 0
fi
echo "V10_STAGE2_OPT_INCOMPLETE"
exit 1
