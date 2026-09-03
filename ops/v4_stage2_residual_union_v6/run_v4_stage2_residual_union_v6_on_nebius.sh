#!/usr/bin/env bash
set -euo pipefail

# Stage2-only residual-union experiment. The output contract is intentionally identical to the
# prior canonical Stage2/Stage3 quality runs:
#   .../v4_stage23_quality/full_run_v2_<UTC>/stage2/<SID>/...
#   .../v4_stage23_quality/full_run_v2_<UTC>/stage3/       (reserved, empty)

EXP_REPO="${EXP_REPO:-/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_residual_union_v6}"
TOOL_DIR="$EXP_REPO/ops/v4_stage2_residual_union_v6"
HOST_OUTPUTS="${HOST_OUTPUTS:-/workspace/voxel_poleline/outputs}"
CONTAINER_OUTPUTS="${CONTAINER_OUTPUTS:-/outputs}"
IMAGE="${IMAGE:-va-v4-realtime:torch241-cu121}"
PRODUCTION_TAG="${PRODUCTION_TAG:-v4.0.1-production-ops}"

BASELINE_RUN_HOST="${BASELINE_RUN_HOST:-$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_production/full_dataset_runs/d9977c39c443f5fa14f8/20260825T203403Z}"
STAGE1_ROOT_HOST="${STAGE1_ROOT_HOST:-$BASELINE_RUN_HOST/stage1}"
STAGE2_BUNDLE_HOST="${STAGE2_BUNDLE_HOST:-$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_realtime/stage2_refiner/local_refiner_bundle.joblib}"
CALIBRATION_HOST="${CALIBRATION_HOST:-$HOST_OUTPUTS/poleline_voxel_run_session_groups/precision_v4/full_val/calibration.json}"

TARGET_GID="${TARGET_GID:-VELASCO_CUT_CP/session1}"
PRECISION_SLICE_MIN="${PRECISION_SLICE_MIN:-0}"
PRECISION_SLICE_MAX="${PRECISION_SLICE_MAX:-20}"
RECALL_SLICE_MIN="${RECALL_SLICE_MIN:-21}"
RECALL_SLICE_MAX="${RECALL_SLICE_MAX:-40}"
WINDOW_MODE="${WINDOW_MODE:-ordinal}"
GUARDRAIL_SLICES_PER_SESSION="${GUARDRAIL_SLICES_PER_SESSION:-2}"
RUN_SCOPE="${RUN_SCOPE:-all}"
MAX_SLICES="${MAX_SLICES:-0}"
RESUME="${RESUME:-0}"
WRITE_VOXEL_AUDIT="${WRITE_VOXEL_AUDIT:-1}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
RUN_STAMP="${RUN_STAMP:-}"
SELECTOR_VERSION_EXPECTED="residual-union-v6-20260903"
SELECTOR_SHA256_EXPECTED="3cb26e581e945182fb47413156c2d3e24aa21eff9f48b49b2a92864e34d87c9f"

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
      status=(s ? $s : "completed")
      gsub(/^"|"$/, "", gid)
      gsub(/^"|"$/, "", status)
      gsub(/\r/, "", gid)
      gsub(/\r/, "", status)
      if ((g==0 || gid==wanted) && (s==0 || status=="completed")) n++
    }
    END { print n+0 }
  ' "$manifest"
}

case "$RUN_SCOPE" in all|target) ;; *) fail "RUN_SCOPE must be all or target" ;; esac
case "$RESUME" in 0|1) ;; *) fail "RESUME must be 0 or 1" ;; esac
case "$WRITE_VOXEL_AUDIT" in 0|1) ;; *) fail "WRITE_VOXEL_AUDIT must be 0 or 1" ;; esac
case "$PREFLIGHT_ONLY" in 0|1) ;; *) fail "PREFLIGHT_ONLY must be 0 or 1" ;; esac
case "$WINDOW_MODE" in ordinal|slice_seq) ;; *) fail "WINDOW_MODE must be ordinal or slice_seq" ;; esac
for value in "$PRECISION_SLICE_MIN" "$PRECISION_SLICE_MAX" "$RECALL_SLICE_MIN" "$RECALL_SLICE_MAX" "$GUARDRAIL_SLICES_PER_SESSION" "$MAX_SLICES"; do
  is_nonnegative_integer "$value" || fail "expected nonnegative integer, got: $value"
done
[ "$PRECISION_SLICE_MIN" -le "$PRECISION_SLICE_MAX" ] || fail "invalid precision window"
[ "$RECALL_SLICE_MIN" -le "$RECALL_SLICE_MAX" ] || fail "invalid recall window"

[ -d "$EXP_REPO/.git" ] || [ -f "$EXP_REPO/.git" ] || fail "experiment Git worktree missing: $EXP_REPO"
[ -d "$TOOL_DIR" ] || fail "experiment tool directory missing: $TOOL_DIR"
for file in \
  v4_stage2_residual_union.py \
  select_v4_stage2_residual_union_profile.py \
  run_v4_stage2_residual_union.py \
  self_test_residual_union.py; do
  [ -f "$TOOL_DIR/$file" ] || fail "required experiment file missing: $TOOL_DIR/$file"
done
[ -d "$STAGE1_ROOT_HOST" ] || fail "saved Stage1 root missing: $STAGE1_ROOT_HOST"
[ -f "$STAGE2_BUNDLE_HOST" ] || fail "accepted Stage2 bundle missing: $STAGE2_BUNDLE_HOST"
[ -f "$CALIBRATION_HOST" ] || fail "accepted calibration missing: $CALIBRATION_HOST"

SELECTOR_HOST="$TOOL_DIR/select_v4_stage2_residual_union_profile.py"
grep -Fq "RESIDUAL_UNION_SELECTOR_VERSION = \"$SELECTOR_VERSION_EXPECTED\"" "$SELECTOR_HOST" \
  || fail "stale selector installed: expected version $SELECTOR_VERSION_EXPECTED"
grep -Fq -- "--window_mode" "$SELECTOR_HOST" \
  || fail "selector lacks --window_mode; stale source installed"
SELECTOR_SHA256_ACTUAL=$(sha256sum "$SELECTOR_HOST" | awk '{print $1}')
[ "$SELECTOR_SHA256_ACTUAL" = "$SELECTOR_SHA256_EXPECTED" ] \
  || fail "selector SHA256 mismatch: expected=$SELECTOR_SHA256_EXPECTED actual=$SELECTOR_SHA256_ACTUAL"

git -C "$EXP_REPO" rev-parse -q --verify "${PRODUCTION_TAG}^{commit}" >/dev/null \
  || fail "production tag unavailable: $PRODUCTION_TAG"
if ! git -C "$EXP_REPO" diff --quiet "$PRODUCTION_TAG" -- v4; then
  fail "production V4 implementation differs from $PRODUCTION_TAG; experiment code must remain under ops only"
fi
if git -C "$EXP_REPO" grep -q 'recover_line_candidates_auto' -- v4 2>/dev/null; then
  fail "obsolete post-refiner recovery hook is present"
fi

STAGE1_ROOT_C=$(host_to_container "$STAGE1_ROOT_HOST")
STAGE2_BUNDLE_C=$(host_to_container "$STAGE2_BUNDLE_HOST")
CALIBRATION_C=$(host_to_container "$CALIBRATION_HOST")

mapfile -t ALL_MANIFESTS < <(
  find "$STAGE1_ROOT_HOST" \
    -mindepth 2 \
    -maxdepth 2 \
    -type f \
    -name stage1_manifest.csv \
    | sort
)
[ "${#ALL_MANIFESTS[@]}" -gt 0 ] || fail "no Stage1 session manifests found"

declare -A SEEN_GID=()
declare -A SEEN_SID=()
TARGET_MANIFEST=""
for manifest in "${ALL_MANIFESTS[@]}"; do
  gid=$(extract_group_id "$manifest")
  [ -n "$gid" ] || fail "group_id missing: $manifest"
  sid=$(basename "$(dirname "$manifest")")
  expected_sid=$(printf '%s' "$gid" | sed -E 's/[^A-Za-z0-9_.-]+/__/g')
  [ "$sid" = "$expected_sid" ] || fail "SID mismatch: actual=$sid expected=$expected_sid"
  [ -z "${SEEN_GID[$gid]:-}" ] || fail "duplicate group_id: $gid"
  [ -z "${SEEN_SID[$sid]:-}" ] || fail "duplicate SID: $sid"
  SEEN_GID[$gid]=1
  SEEN_SID[$sid]=1
  [ "$(completed_count "$manifest" "$gid")" -gt 0 ] || fail "no completed Stage1 rows for $gid"
  if [ "$gid" = "$TARGET_GID" ]; then
    TARGET_MANIFEST="$manifest"
  fi
done
[ -n "$TARGET_MANIFEST" ] || fail "target session not found: $TARGET_GID"
TARGET_COMPLETED_ROWS=$(completed_count "$TARGET_MANIFEST" "$TARGET_GID")
if [ "$WINDOW_MODE" = "ordinal" ]; then
  REQUIRED_ORDINAL_MAX=$PRECISION_SLICE_MAX
  [ "$RECALL_SLICE_MAX" -le "$REQUIRED_ORDINAL_MAX" ] || REQUIRED_ORDINAL_MAX=$RECALL_SLICE_MAX
  REQUIRED_ORDINAL_ROWS=$((REQUIRED_ORDINAL_MAX + 1))
  [ "$TARGET_COMPLETED_ROWS" -ge "$REQUIRED_ORDINAL_ROWS" ] \
    || fail "target session has only $TARGET_COMPLETED_ROWS completed rows; ordinal windows require $REQUIRED_ORDINAL_ROWS"
fi

# Validate exact container paths, bundle, calibration, import graph and core invariants.
docker run --rm -i \
  --mount "type=bind,source=$EXP_REPO/v4,target=/workspace/v4,readonly" \
  --mount "type=bind,source=$TOOL_DIR,target=/workspace/quality,readonly" \
  --mount "type=bind,source=$HOST_OUTPUTS,target=$CONTAINER_OUTPUTS,readonly" \
  --workdir /workspace/v4 \
  -e PYTHONPATH=/workspace/v4:/workspace/quality \
  "$IMAGE" \
  bash -lc "
    set -euo pipefail
    test -d '$STAGE1_ROOT_C'
    test -f '$STAGE2_BUNDLE_C'
    test -f '$CALIBRATION_C'
    python -m py_compile \
      /workspace/quality/v4_stage2_residual_union.py \
      /workspace/quality/select_v4_stage2_residual_union_profile.py \
      /workspace/quality/run_v4_stage2_residual_union.py \
      /workspace/quality/self_test_residual_union.py
    python /workspace/quality/self_test_residual_union.py
    python /workspace/quality/select_v4_stage2_residual_union_profile.py --self_test
    python - '$STAGE2_BUNDLE_C' '$CALIBRATION_C' <<'PY'
import json
import sys
from pathlib import Path
import joblib
bundle_path=Path(sys.argv[1])
calibration_path=Path(sys.argv[2])
bundle=joblib.load(bundle_path)
for key in ('pole_model','line_model','pole_threshold','line_threshold'):
    if key not in bundle:
        raise SystemExit(f'Stage2 bundle missing {key}')
cal=json.loads(calibration_path.read_text())
if not any(key in cal for key in ('pole_threshold','threshold')):
    raise SystemExit('calibration pole threshold missing')
if not any(key in cal for key in ('line_threshold','threshold')):
    raise SystemExit('calibration line threshold missing')
print('STAGE1_LABEL_SOURCE_AND_PATH_PREFLIGHT_OK')
PY
  "

COMMIT=$(git -C "$EXP_REPO" rev-parse HEAD)
if [ -z "$RUN_STAMP" ]; then
  RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
fi
[[ "$RUN_STAMP" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || fail "RUN_STAMP must use YYYYMMDDTHHMMSSZ"

# Fixed canonical output layout. The run-name pattern and all subdirectories are
# the same as the previous canonical Stage2 result.
RUN_HOST="$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_stage23_quality/full_run_v2_${RUN_STAMP}"
RUN_C="$CONTAINER_OUTPUTS/poleline_voxel_run_session_groups/v4_stage23_quality/full_run_v2_${RUN_STAMP}"
STAGE2_HOST="$RUN_HOST/stage2"
STAGE2_C="$RUN_C/stage2"
STAGE3_HOST="$RUN_HOST/stage3"
SELECTION_HOST="$RUN_HOST/selection"
SELECTION_C="$RUN_C/selection"
LOG_HOST="$RUN_HOST/logs"
TIMING_HOST="$RUN_HOST/timing"
STATUS_HOST="$RUN_HOST/status"
SESSION_MAP="$RUN_HOST/session_map.tsv"

if [ "$PREFLIGHT_ONLY" = "1" ]; then
  echo "============================================================"
  echo "V4 STAGE2 RESIDUAL-UNION V6 PREFLIGHT OK"
  echo "============================================================"
  echo "repository=$EXP_REPO"
  echo "commit=$COMMIT"
  echo "stage1_root=$STAGE1_ROOT_HOST"
  echo "selector_version=$SELECTOR_VERSION_EXPECTED"
  echo "selector_sha256=$SELECTOR_SHA256_ACTUAL"
  echo "target_completed_rows=$TARGET_COMPLETED_ROWS"
  echo "window_mode=$WINDOW_MODE"
  echo "stage2_bundle=$STAGE2_BUNDLE_HOST"
  echo "calibration=$CALIBRATION_HOST"
  echo "stage1_sessions=${#ALL_MANIFESTS[@]}"
  echo "target_session=$TARGET_GID"
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

printf '%s\n' "$RUN_HOST" > "$HOME/LATEST_V4_STAGE2_RESIDUAL_UNION_V6_RUN.txt"
printf '%s\n' "$RUN_HOST" > "$HOME/LATEST_V4_STAGE23_QUALITY_V2.txt"

cat > "$RUN_HOST/RUN_INFO.txt" <<EOF
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
layout_contract=v4_stage23_quality/full_run_v2_<UTC>/stage2/<SID>
experiment=production_baseline_plus_stage1_residual_parallel_union_v6
repository=$EXP_REPO
branch=$(git -C "$EXP_REPO" branch --show-current)
commit=$COMMIT
production_base=$PRODUCTION_TAG
stage1_source=$STAGE1_ROOT_HOST
stage1_rerun=false
stage2_bundle=$STAGE2_BUNDLE_HOST
calibration=$CALIBRATION_HOST
target_session=$TARGET_GID
window_mode=$WINDOW_MODE
precision_window=$PRECISION_SLICE_MIN-$PRECISION_SLICE_MAX
recall_window=$RECALL_SLICE_MIN-$RECALL_SLICE_MAX
guardrail_slices_per_session=$GUARDRAIL_SLICES_PER_SESSION
runtime_line_source=production_stage2_plus_deployed_v4_label_residual_class_2
runtime_gt_usage=false
synthetic_line_voxels=0
stage2_scope=$RUN_SCOPE
max_slices=$MAX_SLICES
resume=$RESUME
write_voxel_audit=$WRITE_VOXEL_AUDIT
stage3_ran=false
stage3_directory_reserved=$STAGE3_HOST
EOF

# Offline-only selector. Ground truth is used only here as a correctness guardrail.
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
  python /workspace/quality/select_v4_stage2_residual_union_profile.py \
    --stage1_root "$STAGE1_ROOT_C" \
    --stage2_bundle "$STAGE2_BUNDLE_C" \
    --calibration_json "$CALIBRATION_C" \
    --output_dir "$SELECTION_C" \
    --target_session "$TARGET_GID" \
    --precision_slice_min "$PRECISION_SLICE_MIN" \
    --precision_slice_max "$PRECISION_SLICE_MAX" \
    --recall_slice_min "$RECALL_SLICE_MIN" \
    --recall_slice_max "$RECALL_SLICE_MAX" \
    --window_mode "$WINDOW_MODE" \
    --guardrail_slices_per_session "$GUARDRAIL_SLICES_PER_SESSION" \
  2>&1 | tee "$LOG_HOST/stage2/profile_selection.log"
SELECT_EXIT=${PIPESTATUS[0]}
set -e

if [ "$SELECT_EXIT" -ne 0 ]; then
  if [ "$SELECT_EXIT" -eq 3 ] && [ -f "$SELECTION_HOST/NO_SAFE_RESIDUAL_UNION_PROFILE.txt" ]; then
    cp "$SELECTION_HOST/NO_SAFE_RESIDUAL_UNION_PROFILE.txt" "$RUN_HOST/NO_SAFE_RESIDUAL_UNION_PROFILE.txt"
    find "$RUN_HOST" -type f | sort > "$RUN_HOST/FILE_INVENTORY.txt"
    PACKAGE_HOST="$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_stage23_quality/packages"
    mkdir -p "$PACKAGE_HOST"
    ARCHIVE="$PACKAGE_HOST/v4_stage2_residual_union_selection_full_run_v2_${RUN_STAMP}.tar.gz"
    tar -czf "$ARCHIVE" -C "$RUN_HOST" selection logs RUN_INFO.txt NO_SAFE_RESIDUAL_UNION_PROFILE.txt FILE_INVENTORY.txt
    tar -tzf "$ARCHIVE" >/dev/null
    sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
    printf '%s\n' "$ARCHIVE" > "$HOME/LATEST_V4_STAGE2_RESIDUAL_UNION_V6_ARCHIVE.txt"
    echo "NO_SAFE_RESIDUAL_UNION_PROFILE"
    echo "Diagnostic archive: $ARCHIVE"
    exit 3
  fi
  fail "profile selector failed with exit code $SELECT_EXIT"
fi

[ -f "$SELECTION_HOST/selected_profile.json" ] || fail "selected profile JSON missing"
[ -f "$SELECTION_HOST/selected_profile.env" ] || fail "selected profile environment missing"
[ -f "$SELECTION_HOST/selection_report.txt" ] || fail "selection report missing"
grep -Fq 'SELECTED_SAFE_RESIDUAL_UNION_PROFILE' "$SELECTION_HOST/selection_report.txt" \
  || fail "selector did not approve a safe profile"

if [ "$RUN_SCOPE" = "target" ]; then
  MANIFESTS=("$TARGET_MANIFEST")
else
  MANIFESTS=("${ALL_MANIFESTS[@]}")
fi

: > "$SESSION_MAP"
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
  echo "STAGE2 RESIDUAL-UNION $INDEX/$TOTAL"
  echo "GID=$GID"
  echo "SID=$SID"
  echo "Stage1=$S1_SESSION_C"
  echo "Stage2=$S2_SESSION_C"
  echo "Profile=$SELECTION_C/selected_profile.json"
  echo "============================================================"

  docker run --rm \
    --mount "type=bind,source=$EXP_REPO/v4,target=/workspace/v4,readonly" \
    --mount "type=bind,source=$TOOL_DIR,target=/workspace/quality,readonly" \
    --mount "type=bind,source=$HOST_OUTPUTS,target=$CONTAINER_OUTPUTS" \
    --workdir /workspace/v4 \
    -e PYTHONPATH=/workspace/v4:/workspace/quality \
    -e PYTHONPYCACHEPREFIX=/tmp/pycache \
    -e MPLCONFIGDIR=/tmp/matplotlib \
    "$IMAGE" \
    python /workspace/quality/run_v4_stage2_residual_union.py \
      --stage1_dir "$S1_SESSION_C" \
      --output_dir "$S2_SESSION_C" \
      --session_filter "$GID" \
      --stage2_bundle "$STAGE2_BUNDLE_C" \
      --calibration_json "$CALIBRATION_C" \
      --profile_json "$SELECTION_C/selected_profile.json" \
      --timing_csv "$TIMING_FILE_C" \
      --resume "$RESUME" \
      --max_slices "$MAX_SLICES" \
      --write_voxel_audit "$WRITE_VOXEL_AUDIT" \
    2>&1 | tee "$LOG_FILE"

  [ -f "$S2_SESSION_HOST/STAGE2_COMPLETED.json" ] || fail "Stage2 marker missing for $GID"
  [ -f "$S2_SESSION_HOST/STAGE2_RESIDUAL_UNION_SUMMARY.json" ] || fail "residual-union summary missing for $GID"
  [ -f "$S2_SESSION_HOST/inference_manifest.csv" ] || fail "Stage2 inference manifest missing for $GID"
  [ -s "$TIMING_FILE_HOST" ] || fail "Stage2 timing CSV missing or empty for $GID"

  EXPECTED_ROWS=$(completed_count "$manifest" "$GID")
  if [ "$MAX_SLICES" -gt 0 ] && [ "$EXPECTED_ROWS" -gt "$MAX_SLICES" ]; then
    EXPECTED_ROWS=$MAX_SLICES
  fi
  OUTPUT_ROWS=$(completed_count "$S2_SESSION_HOST/inference_manifest.csv" "$GID")
  [ "$OUTPUT_ROWS" -eq "$EXPECTED_ROWS" ] \
    || fail "Stage2 row-count mismatch for $GID: expected=$EXPECTED_ROWS output=$OUTPUT_ROWS"

  if grep -qiE 'Traceback|IsADirectoryError|FileNotFoundError|RuntimeError|Exception:' "$LOG_FILE"; then
    fail "failure text found in Stage2 log for $GID"
  fi
  if grep -Rqi '"runtime_gt_usage": true\|"synthetic_line_voxels": [1-9]' "$S2_SESSION_HOST" --include='*_stage1_label_audit.json'; then
    fail "runtime invariant violation in $GID"
  fi

  printf 'OK %s gid=%s rows=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$GID" "$OUTPUT_ROWS" \
    > "$STATUS_FILE"
done

DONE=$(find "$STATUS_HOST" -maxdepth 1 -type f -name '*.stage2.ok' | wc -l | tr -d ' ')
MANIFEST_COUNT=$(find "$STAGE2_HOST" -mindepth 2 -maxdepth 2 -type f -name inference_manifest.csv | wc -l | tr -d ' ')
TIMING_COUNT=$(find "$TIMING_HOST/stage2" -maxdepth 1 -type f -name '*.csv' | wc -l | tr -d ' ')
[ "$DONE" -eq "$TOTAL" ] || fail "completion count mismatch: expected=$TOTAL done=$DONE"
[ "$MANIFEST_COUNT" -eq "$TOTAL" ] || fail "manifest count mismatch: expected=$TOTAL manifests=$MANIFEST_COUNT"
[ "$TIMING_COUNT" -eq "$TOTAL" ] || fail "timing count mismatch: expected=$TOTAL timings=$TIMING_COUNT"

printf 'PHASE2_STAGE2_OK %s expected=%s completed=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$TOTAL" "$DONE" \
  > "$RUN_HOST/PHASE2_STAGE2_OK.txt"
printf 'STAGE2_ONLY_COMPLETE %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$RUN_HOST/STAGE2_ONLY_COMPLETE.txt"
find "$RUN_HOST" -type f | sort > "$RUN_HOST/FILE_INVENTORY.txt"

PACKAGE_HOST="$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_stage23_quality/packages"
mkdir -p "$PACKAGE_HOST"
RUN_ID=$(basename "$RUN_HOST")
ARCHIVE="$PACKAGE_HOST/v4_stage2_residual_union_${RUN_ID}.tar.gz"

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
printf '%s\n' "$ARCHIVE" > "$HOME/LATEST_V4_STAGE2_RESIDUAL_UNION_V6_ARCHIVE.txt"
printf '%s\n' "$ARCHIVE" > "$HOME/LATEST_V4_STAGE2_ALL_CANONICAL_ARCHIVE.txt"

cat > "$RUN_HOST/STAGE2_PACKAGE_INFO.txt" <<EOF
archive=$ARCHIVE
archive_sha256=$(awk '{print $1}' "${ARCHIVE}.sha256")
packaged_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

echo "============================================================"
echo "V4 STAGE2 RESIDUAL-UNION CANONICAL RUN COMPLETE"
echo "============================================================"
echo "Run root:        $RUN_HOST"
echo "Stage2 root:     $STAGE2_HOST"
echo "Stage3 reserved: $STAGE3_HOST"
echo "Sessions:        $DONE"
echo "Archive:         $ARCHIVE"
echo "Profile:         $(awk -F= '/^profile=/{print $2}' "$SELECTION_HOST/selection_report.txt" | head -1)"
echo "Runtime GT use:  false"
echo "Synthetic voxels:0"
echo "============================================================"
