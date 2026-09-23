#!/usr/bin/env bash
set -euo pipefail

# Canonical target/all Stage-2 runner for V9 Stage-1 inferred track joining.
# Does NOT rerun Stage 1. Does NOT run Stage 3.
# Canonical output contract:
#   .../v4_stage23_quality/full_run_v2_<UTC>/
#     stage2/<SID>/
#     stage3/                  # reserved, empty
#     selection/               # policy metadata only
#     logs/stage2/
#     logs/stage3/             # reserved, empty
#     timing/stage2/
#     status/
#     RUN_INFO.txt
#     session_map.tsv
#     PHASE2_STAGE2_OK.txt
#     STAGE2_ONLY_COMPLETE.txt
#     FILE_INVENTORY.txt

EXP_REPO="${EXP_REPO:-/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_stage1_track_join_v9}"
TOOL_DIR="$EXP_REPO/ops/v4_stage2_stage1_track_join_v9"
HOST_OUTPUTS="${HOST_OUTPUTS:-/workspace/voxel_poleline/outputs}"
CONTAINER_OUTPUTS="${CONTAINER_OUTPUTS:-/outputs}"
IMAGE="${IMAGE:-va-v4-realtime:torch241-cu121}"
PRODUCTION_TAG="${PRODUCTION_TAG:-v4.0.1-production-ops}"

BASELINE_RUN_HOST="${BASELINE_RUN_HOST:-$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_production/full_dataset_runs/d9977c39c443f5fa14f8/20260825T203403Z}"
STAGE1_ROOT_HOST="${STAGE1_ROOT_HOST:-$BASELINE_RUN_HOST/stage1}"
STAGE2_BUNDLE_HOST="${STAGE2_BUNDLE_HOST:-$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_realtime/stage2_refiner/local_refiner_bundle.joblib}"
CALIBRATION_HOST="${CALIBRATION_HOST:-$HOST_OUTPUTS/poleline_voxel_run_session_groups/precision_v4/full_val/calibration.json}"

RUN_STAMP="${RUN_STAMP:-}"
MAX_SLICES="${MAX_SLICES:-0}"
RESUME="${RESUME:-0}"
WRITE_VOXEL_AUDIT="${WRITE_VOXEL_AUDIT:-1}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
RUN_SCOPE="${RUN_SCOPE:-target}"
TARGET_GID="${TARGET_GID:-VELASCO_CUT_CP/session1}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

is_nonnegative_integer() {
  [[ "${1:-}" =~ ^[0-9]+$ ]]
}

host_to_container() {
  local path="${1:-}"
  [ -n "$path" ] || fail "empty host path"
  case "$path" in
    "$HOST_OUTPUTS") printf '%s\n' "$CONTAINER_OUTPUTS" ;;
    "$HOST_OUTPUTS"/*) printf '%s%s\n' "$CONTAINER_OUTPUTS" "${path#$HOST_OUTPUTS}" ;;
    *) fail "path lies outside mounted output root: $path" ;;
  esac
}

extract_group_id() {
  local manifest="$1"
  awk -F',' '
    NR==1 {
      for (i=1; i<=NF; i++) {
        h=$i
        gsub(/^"|"$/, "", h)
        gsub(/\r/, "", h)
        if (h=="group_id") g=i
      }
      next
    }
    NR>1 && g {
      v=$g
      gsub(/^"|"$/, "", v)
      gsub(/\r/, "", v)
      if (v!="") { print v; exit }
    }
  ' "$manifest"
}

completed_count() {
  local manifest="$1"
  local wanted_gid="$2"
  awk -F',' -v wanted="$wanted_gid" '
    NR==1 {
      for (i=1; i<=NF; i++) {
        h=$i
        gsub(/^"|"$/, "", h)
        gsub(/\r/, "", h)
        if (h=="group_id") g=i
        if (h=="status") s=i
      }
      next
    }
    NR>1 {
      gid=(g ? $g : "")
      st=(s ? $s : "completed")
      gsub(/^"|"$/, "", gid)
      gsub(/^"|"$/, "", st)
      gsub(/\r/, "", gid)
      gsub(/\r/, "", st)
      if ((g==0 || gid==wanted) && (s==0 || st=="completed")) n++
    }
    END { print n+0 }
  ' "$manifest"
}

case "$RESUME" in 0|1) ;; *) fail "RESUME must be 0 or 1" ;; esac
case "$WRITE_VOXEL_AUDIT" in 0|1) ;; *) fail "WRITE_VOXEL_AUDIT must be 0 or 1" ;; esac
case "$PREFLIGHT_ONLY" in 0|1) ;; *) fail "PREFLIGHT_ONLY must be 0 or 1" ;; esac
case "$RUN_SCOPE" in target|all) ;; *) fail "RUN_SCOPE must be target or all" ;; esac
is_nonnegative_integer "$MAX_SLICES" || fail "MAX_SLICES must be a nonnegative integer"

[ -d "$EXP_REPO/.git" ] || [ -f "$EXP_REPO/.git" ] || fail "experiment Git worktree missing: $EXP_REPO"
[ -d "$TOOL_DIR" ] || fail "V9 tool directory missing: $TOOL_DIR"
for f in run_v4_stage2_stage1_track_join.py v4_stage2_stage1_track_join.py learn_velasco_stage1_track_profile.py self_test_stage1_track_join.py; do
  [ -f "$TOOL_DIR/$f" ] || fail "V9 required file missing: $TOOL_DIR/$f"
done
[ -d "$STAGE1_ROOT_HOST" ] || fail "saved Stage1 root missing: $STAGE1_ROOT_HOST"
[ -f "$STAGE2_BUNDLE_HOST" ] || fail "accepted Stage2 bundle missing: $STAGE2_BUNDLE_HOST"
[ -f "$CALIBRATION_HOST" ] || fail "accepted calibration missing: $CALIBRATION_HOST"

git -C "$EXP_REPO" rev-parse -q --verify "${PRODUCTION_TAG}^{commit}" >/dev/null \
  || fail "production tag unavailable: $PRODUCTION_TAG"
if ! git -C "$EXP_REPO" diff --quiet "$PRODUCTION_TAG" -- v4; then
  fail "production V4 implementation differs from $PRODUCTION_TAG; V9 code must remain under ops only"
fi

STAGE1_ROOT_C=$(host_to_container "$STAGE1_ROOT_HOST")
STAGE2_BUNDLE_C=$(host_to_container "$STAGE2_BUNDLE_HOST")
CALIBRATION_C=$(host_to_container "$CALIBRATION_HOST")

mapfile -t ALL_MANIFESTS < <(
  find "$STAGE1_ROOT_HOST" \
    -mindepth 2 \
    -maxdepth 2 \
    -type f \
    -name 'stage1_manifest.csv' \
    | sort
)
[ "${#ALL_MANIFESTS[@]}" -gt 0 ] || fail "no Stage1 session manifests found"

declare -A SEEN_GID=()
declare -A SEEN_SID=()
TOTAL_STAGE1_ROWS=0
TARGET_MANIFEST=""
for manifest in "${ALL_MANIFESTS[@]}"; do
  gid=$(extract_group_id "$manifest")
  [ -n "$gid" ] || fail "group_id missing: $manifest"
  sid=$(basename "$(dirname "$manifest")")
  expected_sid=$(printf '%s' "$gid" | sed -E 's/[^A-Za-z0-9_.-]+/__/g')
  [ "$sid" = "$expected_sid" ] || fail "SID mismatch: actual=$sid expected=$expected_sid manifest=$manifest"
  [ -z "${SEEN_GID[$gid]:-}" ] || fail "duplicate group_id: $gid"
  [ -z "${SEEN_SID[$sid]:-}" ] || fail "duplicate SID: $sid"
  SEEN_GID[$gid]=1
  SEEN_SID[$sid]=1
  n=$(completed_count "$manifest" "$gid")
  [ "$n" -gt 0 ] || fail "no completed Stage1 rows for $gid"
  TOTAL_STAGE1_ROWS=$((TOTAL_STAGE1_ROWS + n))
  if [ "$gid" = "$TARGET_GID" ]; then TARGET_MANIFEST="$manifest"; fi
done
[ -n "$TARGET_MANIFEST" ] || fail "target session not found: $TARGET_GID"

COMMIT=$(git -C "$EXP_REPO" rev-parse HEAD)
if [ -z "$RUN_STAMP" ]; then
  RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
fi
[[ "$RUN_STAMP" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || fail "RUN_STAMP must use YYYYMMDDTHHMMSSZ"

RUN_HOST="$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_stage23_quality/full_run_v2_${RUN_STAMP}"
RUN_C="$CONTAINER_OUTPUTS/poleline_voxel_run_session_groups/v4_stage23_quality/full_run_v2_${RUN_STAMP}"
STAGE2_HOST="$RUN_HOST/stage2"
STAGE2_C="$RUN_C/stage2"
STAGE3_HOST="$RUN_HOST/stage3"
SELECTION_HOST="$RUN_HOST/selection"
LOG_HOST="$RUN_HOST/logs"
TIMING_HOST="$RUN_HOST/timing"
STATUS_HOST="$RUN_HOST/status"
SESSION_MAP="$RUN_HOST/session_map.tsv"

# Real integration preflight without creating canonical output.
docker run --rm \
  --mount "type=bind,source=$EXP_REPO/v4,target=/workspace/v4,readonly" \
  --mount "type=bind,source=$TOOL_DIR,target=/workspace/quality,readonly" \
  --mount "type=bind,source=$HOST_OUTPUTS,target=$CONTAINER_OUTPUTS,readonly" \
  --workdir /workspace/v4 \
  -e PYTHONPATH=/workspace/v4:/workspace/quality \
  -e PYTHONPYCACHEPREFIX=/tmp/pycache \
  "$IMAGE" \
  bash -lc '
    set -euo pipefail
    python -m py_compile \
      /workspace/quality/v4_stage2_stage1_track_join.py \
      /workspace/quality/learn_velasco_stage1_track_profile.py \
      /workspace/quality/run_v4_stage2_stage1_track_join.py \
      /workspace/quality/self_test_stage1_track_join.py
    python /workspace/quality/self_test_stage1_track_join.py
    python -c "from v4_realtime_pipeline import V4Stage2Processor; import v4_stage2_stage1_track_join as t; assert V4Stage2Processor.__module__ == \"v4_realtime_pipeline\"; print(\"V9_TRACK_JOIN_IMPORT_OK\"); print(t.STAGE1_TRACK_JOIN_RUNTIME_VERSION)"
    rm -rf /tmp/v9profile /tmp/v9stage2 /tmp/v9time.csv
    python /workspace/quality/learn_velasco_stage1_track_profile.py \
      --stage1_dir "'$STAGE1_ROOT_C'/VELASCO_CUT_CP__session1" \
      --calibration_json "'$CALIBRATION_C'" \
      --session_filter "'$TARGET_GID'" \
      --reference_min 0 --reference_max 19 \
      --fragment_min 20 --fragment_max 39 \
      --output_dir /tmp/v9profile
    python /workspace/quality/run_v4_stage2_stage1_track_join.py \
      --stage1_dir "'$STAGE1_ROOT_C'/VELASCO_CUT_CP__session1" \
      --output_dir /tmp/v9stage2 \
      --session_filter "'$TARGET_GID'" \
      --stage2_bundle "'$STAGE2_BUNDLE_C'" \
      --calibration_json "'$CALIBRATION_C'" \
      --profile_json /tmp/v9profile/selected_join_profile.json \
      --timing_csv /tmp/v9time.csv \
      --resume 0 --max_slices 1 --write_voxel_audit 1
    test -f /tmp/v9stage2/STAGE2_STAGE1_TRACK_JOIN_SUMMARY.json
    echo V9_STAGE1_TRACK_JOIN_REAL_SLICE_PREFLIGHT_OK
  '


if [ "$PREFLIGHT_ONLY" = "1" ]; then
  echo "============================================================"
  echo "V9 STAGE1 TRACK-JOIN CANONICAL PREFLIGHT OK"
  echo "============================================================"
  echo "repository=$EXP_REPO"
  echo "commit=$COMMIT"
  echo "stage1_root=$STAGE1_ROOT_HOST"
  echo "stage1_sessions=${#ALL_MANIFESTS[@]}"
  echo "run_scope=$RUN_SCOPE"
  echo "target_gid=$TARGET_GID"
  echo "stage1_completed_rows=$TOTAL_STAGE1_ROWS"
  echo "canonical_run_host=$RUN_HOST"
  echo "canonical_stage2_host=$STAGE2_HOST"
  echo "reserved_stage3_host=$STAGE3_HOST"
  echo "============================================================"
  exit 0
fi

[ ! -e "$RUN_HOST" ] || fail "fresh canonical run root already exists: $RUN_HOST"
mkdir -p \
  "$STAGE2_HOST" \
  "$STAGE3_HOST" \
  "$SELECTION_HOST" \
  "$LOG_HOST/stage2" \
  "$LOG_HOST/stage3" \
  "$TIMING_HOST/stage2" \
  "$STATUS_HOST"

printf '%s\n' "$RUN_HOST" > "$HOME/LATEST_V4_STAGE2_STAGE1_TRACK_JOIN_V9_RUN.txt"
printf '%s\n' "$RUN_HOST" > "$HOME/LATEST_V4_STAGE23_QUALITY_V2.txt"
: > "$SESSION_MAP"

# Learn the join geometry without GT.  Ordinals 0-19 teach stable line thickness/
# separation; 20-39 teach the fragmentation-gap scale that V8 could not bridge.
docker run --rm \
  --mount "type=bind,source=$EXP_REPO/v4,target=/workspace/v4,readonly" \
  --mount "type=bind,source=$TOOL_DIR,target=/workspace/quality,readonly" \
  --mount "type=bind,source=$HOST_OUTPUTS,target=$CONTAINER_OUTPUTS" \
  --workdir /workspace/v4 \
  -e PYTHONPATH=/workspace/v4:/workspace/quality \
  "$IMAGE" \
  python /workspace/quality/learn_velasco_stage1_track_profile.py \
    --stage1_dir "$STAGE1_ROOT_C/VELASCO_CUT_CP__session1" \
    --calibration_json "$CALIBRATION_C" \
    --session_filter "$TARGET_GID" \
    --reference_min 0 --reference_max 19 \
    --fragment_min 20 --fragment_max 39 \
    --output_dir "$RUN_C/selection" \
  2>&1 | tee "$LOG_HOST/stage2/track_profile_learning.log"

PROFILE_HOST="$SELECTION_HOST/selected_join_profile.json"
PROFILE_C="$RUN_C/selection/selected_join_profile.json"
[ -f "$PROFILE_HOST" ] || fail "learned join profile missing"

cat > "$RUN_HOST/RUN_INFO.txt" <<EOF2
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
layout_contract=v4_stage23_quality/full_run_v2_<UTC>/stage2/<SID>
experiment=stage1_inferred_track_join_v9
repository=$EXP_REPO
branch=$(git -C "$EXP_REPO" branch --show-current)
commit=$COMMIT
production_base=$PRODUCTION_TAG
stage1_source=$STAGE1_ROOT_HOST
stage1_rerun=false
stage2_bundle=$STAGE2_BUNDLE_HOST
calibration=$CALIBRATION_HOST
runtime_line_source=deployed_v4_stage1_class_2_track_join
runtime_gt_usage=false
synthetic_line_voxels=0
pole_pair_inference=false
line_hysteresis_used=false
line_refiner_used=false
fragment_bridge_source=stage1_class2_endpoints_only
join_profile=$PROFILE_HOST
stage2_scope=$RUN_SCOPE
stage1_session_count=${#ALL_MANIFESTS[@]}
stage1_completed_rows=$TOTAL_STAGE1_ROWS
max_slices=$MAX_SLICES
resume=$RESUME
write_voxel_audit=$WRITE_VOXEL_AUDIT
stage3_ran=false
stage3_directory_reserved=$STAGE3_HOST
EOF2

if [ "$RUN_SCOPE" = "target" ]; then
  MANIFESTS=("$TARGET_MANIFEST")
else
  MANIFESTS=("${ALL_MANIFESTS[@]}")
fi
TOTAL=${#MANIFESTS[@]}
INDEX=0
for manifest in "${MANIFESTS[@]}"; do
  INDEX=$((INDEX + 1))
  GID=$(extract_group_id "$manifest")
  S1_SESSION_HOST=$(dirname "$manifest")
  SID=$(basename "$S1_SESSION_HOST")
  S1_SESSION_C=$(host_to_container "$S1_SESSION_HOST")
  S2_SESSION_HOST="$STAGE2_HOST/$SID"
  S2_SESSION_C="$STAGE2_C/$SID"
  TIMING_FILE_HOST="$TIMING_HOST/stage2/${SID}.csv"
  TIMING_FILE_C="$RUN_C/timing/stage2/${SID}.csv"
  LOG_FILE="$LOG_HOST/stage2/${SID}.log"
  STATUS_FILE="$STATUS_HOST/${SID}.stage2.ok"

  mkdir -p "$S2_SESSION_HOST"
  printf '%s\t%s\n' "$GID" "$SID" >> "$SESSION_MAP"

  echo "============================================================"
  echo "STAGE2 STAGE1-TRACK $INDEX/$TOTAL"
  echo "GID=$GID"
  echo "SID=$SID"
  echo "Stage1=$S1_SESSION_C"
  echo "Stage2=$S2_SESSION_C"
  echo "Timing=$TIMING_FILE_C"
  echo "============================================================"

  set +e
  docker run --rm \
    --mount "type=bind,source=$EXP_REPO/v4,target=/workspace/v4,readonly" \
    --mount "type=bind,source=$TOOL_DIR,target=/workspace/quality,readonly" \
    --mount "type=bind,source=$HOST_OUTPUTS,target=$CONTAINER_OUTPUTS" \
    --workdir /workspace/v4 \
    -e PYTHONPATH=/workspace/v4:/workspace/quality \
    -e PYTHONPYCACHEPREFIX=/tmp/pycache \
    -e MPLCONFIGDIR=/tmp/matplotlib \
    "$IMAGE" \
    python /workspace/quality/run_v4_stage2_stage1_track_join.py \
      --stage1_dir "$S1_SESSION_C" \
      --output_dir "$S2_SESSION_C" \
      --session_filter "$GID" \
      --stage2_bundle "$STAGE2_BUNDLE_C" \
      --calibration_json "$CALIBRATION_C" \
      --profile_json "$PROFILE_C" \
      --timing_csv "$TIMING_FILE_C" \
      --resume "$RESUME" \
      --max_slices "$MAX_SLICES" \
      --write_voxel_audit "$WRITE_VOXEL_AUDIT" \
    2>&1 | tee "$LOG_FILE"
  SESSION_RC=${PIPESTATUS[0]}
  set -e
  [ "$SESSION_RC" -eq 0 ] || fail "Stage2 track-join run failed for $GID rc=$SESSION_RC"

  [ -f "$S2_SESSION_HOST/STAGE2_COMPLETED.json" ] || fail "Stage2 completion marker missing for $GID"
  [ -f "$S2_SESSION_HOST/STAGE2_STAGE1_TRACK_JOIN_SUMMARY.json" ] || fail "Stage1-track summary missing for $GID"
  [ -f "$S2_SESSION_HOST/inference_manifest.csv" ] || fail "Stage2 inference manifest missing for $GID"
  [ -s "$TIMING_FILE_HOST" ] || fail "Stage2 timing CSV missing/empty for $GID"

  EXPECTED_ROWS=$(completed_count "$manifest" "$GID")
  if [ "$MAX_SLICES" -gt 0 ] && [ "$EXPECTED_ROWS" -gt "$MAX_SLICES" ]; then
    EXPECTED_ROWS=$MAX_SLICES
  fi
  OUTPUT_ROWS=$(completed_count "$S2_SESSION_HOST/inference_manifest.csv" "$GID")
  [ "$OUTPUT_ROWS" -eq "$EXPECTED_ROWS" ] \
    || fail "Stage2 row-count mismatch for $GID: expected=$EXPECTED_ROWS output=$OUTPUT_ROWS"

  if grep -qiE 'Traceback|IsADirectoryError|FileNotFoundError|ImportError|RuntimeError|Exception:' "$LOG_FILE"; then
    fail "failure text found in Stage2 log for $GID"
  fi

  # Verify every emitted per-slice audit preserves Stage1 class-2 voxels exactly.
  docker run --rm -i \
    --mount "type=bind,source=$S2_SESSION_HOST,target=/review,readonly" \
    "$IMAGE" \
    python - <<'PY'
import glob
import json
import math

files = sorted(glob.glob('/review/**/*_stage1_track_join_audit.json', recursive=True))
if not files:
    raise SystemExit('no *_stage1_track_join_audit.json files found')
for path in files:
    with open(path) as f:
        d = json.load(f)
    n1 = int(d['stage1_inferred_line_voxels'])
    n2 = int(d['accepted_stage1_line_voxels'])
    if n1 != n2:
        raise SystemExit(f'voxel preservation mismatch: {path}: {n1} != {n2}')
    p = float(d['stage1_to_stage2_voxel_preservation'])
    if not math.isclose(p, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(f'preservation != 1.0: {path}: {p}')
    if int(d.get('synthetic_line_voxels', 0)) != 0:
        raise SystemExit(f'synthetic line voxels detected: {path}')
    if bool(d.get('runtime_gt_usage', False)):
        raise SystemExit(f'runtime GT usage detected: {path}')
    if bool(d.get('pole_pair_inference', False)):
        raise SystemExit(f'pole-pair inference detected: {path}')
print(f'STAGE1_TRACK_JOIN_SESSION_AUDIT_OK slices={len(files)}')
PY

  printf 'OK %s gid=%s rows=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$GID" "$OUTPUT_ROWS" \
    > "$STATUS_FILE"
done

DONE=$(find "$STATUS_HOST" -maxdepth 1 -type f -name '*.stage2.ok' | wc -l | tr -d ' ')
MANIFEST_COUNT=$(find "$STAGE2_HOST" -mindepth 2 -maxdepth 2 -type f -name inference_manifest.csv | wc -l | tr -d ' ')
SUMMARY_COUNT=$(find "$STAGE2_HOST" -mindepth 2 -maxdepth 2 -type f -name STAGE2_STAGE1_TRACK_JOIN_SUMMARY.json | wc -l | tr -d ' ')
TIMING_COUNT=$(find "$TIMING_HOST/stage2" -maxdepth 1 -type f -name '*.csv' | wc -l | tr -d ' ')

[ "$DONE" -eq "$TOTAL" ] || fail "completion count mismatch: expected=$TOTAL done=$DONE"
[ "$MANIFEST_COUNT" -eq "$TOTAL" ] || fail "manifest count mismatch: expected=$TOTAL manifests=$MANIFEST_COUNT"
[ "$SUMMARY_COUNT" -eq "$TOTAL" ] || fail "summary count mismatch: expected=$TOTAL summaries=$SUMMARY_COUNT"
[ "$TIMING_COUNT" -eq "$TOTAL" ] || fail "timing count mismatch: expected=$TOTAL timings=$TIMING_COUNT"

# Canonical structure guard: direct children under stage2 must be only the expected SIDs.
ACTUAL_SIDS_FILE="$RUN_HOST/.actual_stage2_sids.txt"
EXPECTED_SIDS_FILE="$RUN_HOST/.expected_stage2_sids.txt"
if [ "$RUN_SCOPE" = "target" ]; then
  basename "$(dirname "$TARGET_MANIFEST")" > "$EXPECTED_SIDS_FILE"
else
  printf '%s\n' "${!SEEN_SID[@]}" | sort > "$EXPECTED_SIDS_FILE"
fi
find "$STAGE2_HOST" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort > "$ACTUAL_SIDS_FILE"
cmp "$EXPECTED_SIDS_FILE" "$ACTUAL_SIDS_FILE" \
  || fail "canonical stage2/<SID> directory set mismatch"
rm -f "$EXPECTED_SIDS_FILE" "$ACTUAL_SIDS_FILE"

printf 'PHASE2_STAGE2_OK %s expected=%s completed=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$TOTAL" "$DONE" \
  > "$RUN_HOST/PHASE2_STAGE2_OK.txt"
printf 'STAGE2_ONLY_COMPLETE %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$RUN_HOST/STAGE2_ONLY_COMPLETE.txt"
find "$RUN_HOST" -type f | sort > "$RUN_HOST/FILE_INVENTORY.txt"

PACKAGE_HOST="$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_stage23_quality/packages"
mkdir -p "$PACKAGE_HOST"
RUN_ID=$(basename "$RUN_HOST")
ARCHIVE="$PACKAGE_HOST/v4_stage2_stage1_track_join_${RUN_ID}.tar.gz"

# Package the canonical run structure only. Do not include Stage1, models, bundle, or checkpoints.
tar -czf "$ARCHIVE" \
  --exclude='*.npz' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.ckpt' \
  --exclude='*.onnx' \
  --exclude='*.engine' \
  --exclude='*.safetensors' \
  --exclude='*.joblib' \
  -C "$RUN_HOST" \
  stage2 \
  stage3 \
  selection \
  logs \
  timing \
  status \
  RUN_INFO.txt \
  session_map.tsv \
  PHASE2_STAGE2_OK.txt \
  STAGE2_ONLY_COMPLETE.txt \
  FILE_INVENTORY.txt

tar -tzf "$ARCHIVE" >/dev/null
if tar -tzf "$ARCHIVE" | grep -Ei '\.(npz|pt|pth|ckpt|onnx|engine|safetensors|joblib)$' >/dev/null; then
  fail "excluded model/cache artifact found in archive"
fi
sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
printf '%s\n' "$ARCHIVE" > "$HOME/LATEST_V4_STAGE2_STAGE1_TRACK_JOIN_V9_ARCHIVE.txt"
printf '%s\n' "$ARCHIVE" > "$HOME/LATEST_V4_STAGE2_ALL_CANONICAL_ARCHIVE.txt"

cat > "$RUN_HOST/STAGE2_PACKAGE_INFO.txt" <<EOF2
archive=$ARCHIVE
archive_sha256=$(awk '{print $1}' "${ARCHIVE}.sha256")
packaged_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF2

echo "============================================================"
echo "V4 STAGE2 STAGE1-TRACK CANONICAL RUN COMPLETE"
echo "============================================================"
echo "Run root:        $RUN_HOST"
echo "Stage2 root:     $STAGE2_HOST"
echo "Stage3 reserved: $STAGE3_HOST"
echo "Sessions:        $DONE"
echo "Archive:         $ARCHIVE"
echo "Runtime GT use:  false"
echo "Synthetic voxels:0"
echo "Pole-pair infer: false"
echo "============================================================"
