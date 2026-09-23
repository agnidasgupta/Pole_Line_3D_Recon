#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-va-v4-realtime:torch241-cu121}"
EXP_REPO="${EXP_REPO:-/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_stage1_exact_v8}"
TOOL_DIR="$EXP_REPO/ops/v4_stage2_stage1_exact_v8"
HOST_OUTPUTS="/workspace/voxel_poleline/outputs"
CONTAINER_OUTPUTS="/outputs"

STAGE1_ROOT_HOST="${STAGE1_ROOT_HOST:-$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_production/full_dataset_runs/d9977c39c443f5fa14f8/20260825T203403Z/stage1}"
STAGE2_BUNDLE_HOST="${STAGE2_BUNDLE_HOST:-$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_realtime/stage2_refiner/local_refiner_bundle.joblib}"
CALIBRATION_HOST="${CALIBRATION_HOST:-$HOST_OUTPUTS/poleline_voxel_run_session_groups/precision_v4/full_val/calibration.json}"
OUTPUT_BASE_HOST="$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_stage23_quality"
PACKAGE_HOST="$OUTPUT_BASE_HOST/packages"

RUN_SCOPE="${RUN_SCOPE:-target}"
TARGET_GID="${TARGET_GID:-VELASCO_CUT_CP/session1}"
MAX_SLICES="${MAX_SLICES:-0}"
RESUME="${RESUME:-0}"
WRITE_VOXEL_AUDIT="${WRITE_VOXEL_AUDIT:-1}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

for f in v4_stage2_stage1_exact.py run_v4_stage2_stage1_exact.py self_test_stage1_exact.py; do
  test -f "$TOOL_DIR/$f" || { echo "ERROR: missing $TOOL_DIR/$f"; exit 1; }
done

test -d "$EXP_REPO/.git" || { echo "ERROR: missing experiment Git repo: $EXP_REPO"; exit 1; }
test -d "$STAGE1_ROOT_HOST" || { echo "ERROR: missing Stage1 root: $STAGE1_ROOT_HOST"; exit 1; }
test -f "$STAGE2_BUNDLE_HOST" || { echo "ERROR: missing Stage2 bundle: $STAGE2_BUNDLE_HOST"; exit 1; }
test -f "$CALIBRATION_HOST" || { echo "ERROR: missing calibration: $CALIBRATION_HOST"; exit 1; }
mkdir -p "$OUTPUT_BASE_HOST" "$PACKAGE_HOST"

# Production V4 must remain unchanged on this experiment branch.
if ! git -C "$EXP_REPO" diff --quiet v4.0.1-production-ops -- v4; then
  echo "ERROR: production v4/ differs from v4.0.1-production-ops"
  exit 1
fi

map_output_path() {
  local p="$1"
  case "$p" in
    "$HOST_OUTPUTS"/*) printf '%s%s\n' "$CONTAINER_OUTPUTS" "${p#$HOST_OUTPUTS}" ;;
    *) echo "ERROR: output path outside $HOST_OUTPUTS: $p" >&2; return 1 ;;
  esac
}

STAGE1_ROOT_C=$(map_output_path "$STAGE1_ROOT_HOST")
STAGE2_BUNDLE_C=$(map_output_path "$STAGE2_BUNDLE_HOST")
CALIBRATION_C=$(map_output_path "$CALIBRATION_HOST")

# Verify the exact-mask implementation before any real data are touched.
docker run --rm \
  --mount "type=bind,source=$EXP_REPO/v4,target=/workspace/v4,readonly" \
  --mount "type=bind,source=$TOOL_DIR,target=/workspace/quality,readonly" \
  --workdir /workspace/v4 \
  -e PYTHONPATH=/workspace/v4:/workspace/quality \
  "$IMAGE" \
  bash -lc '
    set -euo pipefail
    python -m py_compile \
      /workspace/quality/v4_stage2_stage1_exact.py \
      /workspace/quality/run_v4_stage2_stage1_exact.py \
      /workspace/quality/self_test_stage1_exact.py
    python /workspace/quality/self_test_stage1_exact.py
  '

# Validate mounts and session inventory in the same runtime namespace.
docker run --rm \
  --mount "type=bind,source=$EXP_REPO/v4,target=/workspace/v4,readonly" \
  --mount "type=bind,source=$HOST_OUTPUTS,target=/outputs" \
  --workdir /workspace/v4 \
  "$IMAGE" \
  bash -lc "
    set -euo pipefail
    test -d '$STAGE1_ROOT_C'
    test -f '$STAGE2_BUNDLE_C'
    test -f '$CALIBRATION_C'
    n=\$(find '$STAGE1_ROOT_C' -mindepth 2 -maxdepth 2 -type f -name stage1_manifest.csv | wc -l)
    test \"\$n\" -eq 30
    echo STAGE1_EXACT_SOURCE_AND_PATH_PREFLIGHT_OK
  "

# Integration smoke: instantiate the real production Stage2 processor and
# process one saved target slice before any canonical run directory is created.
# This catches import/API/runtime mismatches that py_compile cannot detect.
SMOKE_SID="${TARGET_GID//\//__}"
SMOKE_S1_HOST="$STAGE1_ROOT_HOST/$SMOKE_SID"
test -f "$SMOKE_S1_HOST/stage1_manifest.csv" || {
  echo "ERROR: target Stage1 manifest missing for integration smoke: $SMOKE_S1_HOST/stage1_manifest.csv"
  exit 1
}
SMOKE_ROOT_HOST="$OUTPUT_BASE_HOST/.stage1_exact_v8_1_preflight_$$"
SMOKE_OUT_HOST="$SMOKE_ROOT_HOST/stage2/$SMOKE_SID"
SMOKE_TIMING_HOST="$SMOKE_ROOT_HOST/timing/$SMOKE_SID.csv"
mkdir -p "$SMOKE_OUT_HOST" "$(dirname "$SMOKE_TIMING_HOST")"
SMOKE_S1_C=$(map_output_path "$SMOKE_S1_HOST")
SMOKE_OUT_C=$(map_output_path "$SMOKE_OUT_HOST")
SMOKE_TIMING_C=$(map_output_path "$SMOKE_TIMING_HOST")
cleanup_smoke() {
  # The integration-smoke container runs as root and therefore creates
  # root-owned files on the bind-mounted host output tree. Delete those
  # files from a short-lived root container, then remove the empty host
  # directory as the calling user. Cleanup must never turn a successful
  # integration smoke into a failed preflight.
  if [ -d "$SMOKE_ROOT_HOST" ]; then
    docker run --rm \
      --mount "type=bind,source=$SMOKE_ROOT_HOST,target=/smoke" \
      "$IMAGE" \
      bash -lc 'find /smoke -mindepth 1 -delete' \
      >/dev/null 2>&1 || true
    rmdir "$SMOKE_ROOT_HOST" 2>/dev/null || true
  fi
  return 0
}
trap cleanup_smoke EXIT

docker run --rm \
  --mount "type=bind,source=$EXP_REPO/v4,target=/workspace/v4,readonly" \
  --mount "type=bind,source=$TOOL_DIR,target=/workspace/quality,readonly" \
  --mount "type=bind,source=$HOST_OUTPUTS,target=/outputs" \
  --workdir /workspace/v4 \
  -e PYTHONPATH=/workspace/v4:/workspace/quality \
  "$IMAGE" \
  python /workspace/quality/run_v4_stage2_stage1_exact.py \
    --stage1_dir "$SMOKE_S1_C" \
    --output_dir "$SMOKE_OUT_C" \
    --session_filter "$TARGET_GID" \
    --stage2_bundle "$STAGE2_BUNDLE_C" \
    --calibration_json "$CALIBRATION_C" \
    --timing_csv "$SMOKE_TIMING_C" \
    --resume 0 \
    --max_slices 1 \
    --write_voxel_audit 0

test -f "$SMOKE_OUT_HOST/STAGE2_STAGE1_EXACT_SUMMARY.json" || {
  echo "ERROR: integration smoke did not create Stage2 summary"
  exit 1
}

docker run --rm \
  --mount "type=bind,source=$SMOKE_OUT_HOST,target=/smoke,readonly" \
  "$IMAGE" \
  python -c 'import json; d=json.load(open("/smoke/STAGE2_STAGE1_EXACT_SUMMARY.json")); t=d["totals"]; assert t["stage1_inferred_line_voxels"]==t["accepted_stage1_line_voxels"]; assert abs(t["stage1_to_stage2_voxel_preservation"]-1.0)<1e-12; assert d["runtime_gt_usage"] is False; assert d["synthetic_line_voxels"]==0; assert d["pole_pair_inference"] is False; print("STAGE1_EXACT_REAL_SLICE_INTEGRATION_OK")'

cleanup_smoke
trap - EXIT

echo "runtime_version=stage1-exact-v8.1-runtimefix-20260904"
echo "runtime_gt_usage=false"
echo "pole_pair_inference=false"
echo "line_hysteresis_used=false"
echo "line_refiner_used=false"
echo "line_geometry_source=adjacent_stage1_inferred_class2_voxel_pairs"
echo "V4 STAGE2 STAGE1-EXACT V8 PREFLIGHT OK"

if [ "$PREFLIGHT_ONLY" = "1" ]; then
  exit 0
fi

RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ID="full_run_v2_${RUN_STAMP}"
RUN_HOST="$OUTPUT_BASE_HOST/$RUN_ID"
RUN_C=$(map_output_path "$RUN_HOST")

mkdir -p \
  "$RUN_HOST/stage2" \
  "$RUN_HOST/stage3" \
  "$RUN_HOST/selection" \
  "$RUN_HOST/logs/stage2" \
  "$RUN_HOST/timing/stage2" \
  "$RUN_HOST/status"

printf '%s\n' "$RUN_HOST" > "$HOME/LATEST_V4_STAGE2_STAGE1_EXACT_V8_RUN.txt"
printf '%s\n' 'stage1-exact-v8.1-runtimefix-20260904' > "$RUN_HOST/selection/MODE.txt"

cat > "$RUN_HOST/RUN_INFO.txt" <<EOF
experiment=stage1_exact_line_passthrough_v8
runtime_version=stage1-exact-v8.1-runtimefix-20260904
run_scope=$RUN_SCOPE
target_gid=$TARGET_GID
source_commit=$(git -C "$EXP_REPO" rev-parse HEAD)
stage1_root=$STAGE1_ROOT_HOST
stage2_bundle=$STAGE2_BUNDLE_HOST
calibration=$CALIBRATION_HOST
runtime_gt_usage=false
pole_pair_inference=false
line_hysteresis_used=false
line_refiner_used=false
line_geometry_source=adjacent_stage1_inferred_class2_voxel_pairs
canonical_run_root=$RUN_HOST
EOF

# Build session map from the accepted Stage1 directory names. The established
# V4 convention encodes group_id '/' as '__' in the session directory name.
: > "$RUN_HOST/session_map.tsv"
while IFS= read -r manifest; do
  sid=$(basename "$(dirname "$manifest")")
  gid="${sid/__//}"
  if [ "$RUN_SCOPE" = "target" ] && [ "$gid" != "$TARGET_GID" ]; then
    continue
  fi
  printf '%s\t%s\t%s\n' "$gid" "$sid" "$manifest" >> "$RUN_HOST/session_map.tsv"
done < <(find "$STAGE1_ROOT_HOST" -mindepth 2 -maxdepth 2 -type f -name stage1_manifest.csv | sort)

TOTAL=$(wc -l < "$RUN_HOST/session_map.tsv" | tr -d ' ')
if [ "$RUN_SCOPE" = "target" ]; then
  test "$TOTAL" -eq 1 || { echo "ERROR: target session discovery produced $TOTAL sessions"; cat "$RUN_HOST/session_map.tsv"; exit 1; }
else
  test "$TOTAL" -eq 30 || { echo "ERROR: all-data discovery produced $TOTAL sessions"; exit 1; }
fi

INDEX=0
while IFS=$'\t' read -r gid sid manifest; do
  INDEX=$((INDEX + 1))
  S1_HOST=$(dirname "$manifest")
  S1_C=$(map_output_path "$S1_HOST")
  S2_HOST="$RUN_HOST/stage2/$sid"
  S2_C=$(map_output_path "$S2_HOST")
  TIMING_HOST="$RUN_HOST/timing/stage2/$sid.csv"
  TIMING_C=$(map_output_path "$TIMING_HOST")
  LOG_HOST="$RUN_HOST/logs/stage2/$sid.log"
  mkdir -p "$S2_HOST" "$(dirname "$TIMING_HOST")" "$(dirname "$LOG_HOST")"
  echo "[stage2-exact $INDEX/$TOTAL] $gid"

  set +e
  docker run --rm \
    --mount "type=bind,source=$EXP_REPO/v4,target=/workspace/v4,readonly" \
    --mount "type=bind,source=$TOOL_DIR,target=/workspace/quality,readonly" \
    --mount "type=bind,source=$HOST_OUTPUTS,target=/outputs" \
    --workdir /workspace/v4 \
    -e PYTHONPATH=/workspace/v4:/workspace/quality \
    "$IMAGE" \
    python /workspace/quality/run_v4_stage2_stage1_exact.py \
      --stage1_dir "$S1_C" \
      --output_dir "$S2_C" \
      --session_filter "$gid" \
      --stage2_bundle "$STAGE2_BUNDLE_C" \
      --calibration_json "$CALIBRATION_C" \
      --timing_csv "$TIMING_C" \
      --resume "$RESUME" \
      --max_slices "$MAX_SLICES" \
      --write_voxel_audit "$WRITE_VOXEL_AUDIT" \
      > "$LOG_HOST" 2>&1
  rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "ERROR: Stage2 exact run failed for $gid rc=$rc"
    tail -100 "$LOG_HOST"
    exit "$rc"
  fi

  test -f "$S2_HOST/STAGE2_COMPLETED.json" || { echo "ERROR: missing STAGE2_COMPLETED.json for $gid"; exit 1; }
  # Enforce exact preservation invariant from per-session summary.
  docker run --rm \
    --mount "type=bind,source=$S2_HOST,target=/session,readonly" \
    "$IMAGE" \
    python -c 'import json; d=json.load(open("/session/STAGE2_STAGE1_EXACT_SUMMARY.json")); t=d["totals"]; assert t["stage1_inferred_line_voxels"]==t["accepted_stage1_line_voxels"]; assert abs(t["stage1_to_stage2_voxel_preservation"]-1.0)<1e-12; assert d["runtime_gt_usage"] is False; assert d["synthetic_line_voxels"]==0; assert d["pole_pair_inference"] is False; print("SESSION_EXACT_PRESERVATION_OK")'

  printf '%s\n' "completed gid=$gid sid=$sid" > "$RUN_HOST/status/$sid.stage2.ok"
done < "$RUN_HOST/session_map.tsv"

DONE=$(find "$RUN_HOST/status" -maxdepth 1 -type f -name '*.stage2.ok' | wc -l | tr -d ' ')
test "$DONE" -eq "$TOTAL" || { echo "ERROR: completed session count $DONE != $TOTAL"; exit 1; }

printf '%s\n' "PHASE2_STAGE2_OK sessions=$TOTAL runtime=stage1-exact-v8.1-runtimefix-20260904" > "$RUN_HOST/PHASE2_STAGE2_OK.txt"
printf '%s\n' "STAGE2_ONLY_COMPLETE sessions=$TOTAL runtime=stage1-exact-v8.1-runtimefix-20260904" > "$RUN_HOST/STAGE2_ONLY_COMPLETE.txt"

ARCHIVE="$PACKAGE_HOST/v4_stage2_stage1_exact_${RUN_ID}.tar.gz"
rm -f "$ARCHIVE" "${ARCHIVE}.sha256"
(
  cd "$RUN_HOST"
  tar -czf "$ARCHIVE" \
    --exclude='*.npz' \
    --exclude='*.joblib' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='*.ckpt' \
    stage2 stage3 selection logs timing status \
    RUN_INFO.txt session_map.tsv PHASE2_STAGE2_OK.txt STAGE2_ONLY_COMPLETE.txt
)
sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
tar -tzf "$ARCHIVE" >/dev/null
printf '%s\n' "$ARCHIVE" > "$HOME/LATEST_V4_STAGE2_STAGE1_EXACT_V8_ARCHIVE.txt"

echo "============================================================"
echo "V4 STAGE2 STAGE1-EXACT V8 RUN COMPLETE"
echo "============================================================"
echo "Run root:    $RUN_HOST"
echo "Stage2 root: $RUN_HOST/stage2"
echo "Stage3:      $RUN_HOST/stage3 (reserved; empty)"
echo "Sessions:    $TOTAL"
echo "Archive:     $ARCHIVE"
