#!/usr/bin/env bash
set -euo pipefail

# Run V4 Stage 2 for every saved Stage 1 session using the already selected
# GT-validated scalar operating profile. Output layout intentionally matches
# the prior combined Stage2/Stage3 quality run:
#   .../v4_stage23_quality/full_run_v2_<UTC>/stage2/<SID>/...
# with stage3/, logs/, timing/, status/, RUN_INFO.txt and session_map.tsv
# reserved in the same locations for later Stage 3 execution.

EXP_REPO="${EXP_REPO:-/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_gt_autoselect_v3}"
HOST_OUTPUTS="${HOST_OUTPUTS:-/workspace/voxel_poleline/outputs}"
CONTAINER_OUTPUTS="${CONTAINER_OUTPUTS:-/outputs}"
IMAGE="${IMAGE:-va-v4-realtime:torch241-cu121}"
PRODUCTION_TAG="${PRODUCTION_TAG:-v4.0.1-production-ops}"

BASELINE_RUN_HOST="${BASELINE_RUN_HOST:-$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_production/full_dataset_runs/d9977c39c443f5fa14f8/20260825T203403Z}"
STAGE1_ROOT_HOST="${STAGE1_ROOT_HOST:-$BASELINE_RUN_HOST/stage1}"
ORIGINAL_STAGE2_BUNDLE_HOST="${ORIGINAL_STAGE2_BUNDLE_HOST:-$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_realtime/stage2_refiner/local_refiner_bundle.joblib}"

SOURCE_SELECTION_RUN_HOST="${SOURCE_SELECTION_RUN_HOST:-}"
RUN_STAMP="${RUN_STAMP:-}"
MAX_SLICES="${MAX_SLICES:-0}"
RESUME="${RESUME:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

POLE_CANDIDATE_THRESHOLD="${POLE_CANDIDATE_THRESHOLD:-0.15}"
POLE_MIN_VOXELS="${POLE_MIN_VOXELS:-4}"
LINE_MIN_VOXELS="${LINE_MIN_VOXELS:-3}"
EDGE_WIDTH_VOX="${EDGE_WIDTH_VOX:-10}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

host_to_container() {
  local p="${1:-}"
  [ -n "$p" ] || fail "empty host path"
  case "$p" in
    "$HOST_OUTPUTS"/*)
      printf '%s%s\n' "$CONTAINER_OUTPUTS" "${p#$HOST_OUTPUTS}"
      ;;
    *)
      fail "path is outside mounted output root: $p"
      ;;
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
        if (h=="group_id") c=i
      }
      next
    }
    NR>1 && c {
      v=$c
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

is_nonnegative_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

is_number() {
  [[ "$1" =~ ^-?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]
}

case "$RESUME" in 0|1) ;; *) fail "RESUME must be 0 or 1" ;; esac
case "$PREFLIGHT_ONLY" in 0|1) ;; *) fail "PREFLIGHT_ONLY must be 0 or 1" ;; esac
is_nonnegative_integer "$MAX_SLICES" || fail "MAX_SLICES must be a nonnegative integer"

[ -d "$EXP_REPO/.git" ] || [ -f "$EXP_REPO/.git" ] || fail "experiment Git worktree missing: $EXP_REPO"
[ -f "$EXP_REPO/v4/run_v4_stage2.py" ] || fail "Stage2 entry point missing"
[ -f "$EXP_REPO/ops/v4_full_dataset/profile_stage2_session.py" ] || fail "profile_stage2_session.py missing"
[ -d "$STAGE1_ROOT_HOST" ] || fail "saved Stage1 root missing: $STAGE1_ROOT_HOST"
[ -f "$ORIGINAL_STAGE2_BUNDLE_HOST" ] || fail "accepted original Stage2 bundle missing: $ORIGINAL_STAGE2_BUNDLE_HOST"

git -C "$EXP_REPO" rev-parse -q --verify "${PRODUCTION_TAG}^{commit}" >/dev/null \
  || fail "production tag unavailable in experiment repository: $PRODUCTION_TAG"

if git -C "$EXP_REPO" grep -q 'recover_line_candidates_auto' -- v4 2>/dev/null; then
  fail "obsolete post-refiner recovery hook is present"
fi

if ! git -C "$EXP_REPO" diff --quiet "$PRODUCTION_TAG" -- \
    v4/run_v4_stage2.py \
    v4/v4_realtime_pipeline.py \
    v4/v4_sparse_components.py \
    v4/v4_stage2_runtime.py \
    v4/v4_stage2_local.py; then
  fail "core Stage2 source differs from $PRODUCTION_TAG"
fi

if [ -z "$SOURCE_SELECTION_RUN_HOST" ]; then
  POINTER="$HOME/LATEST_V4_STAGE2_GT_AUTOSELECT_V3_RUN.txt"
  [ -f "$POINTER" ] || fail "selection-run pointer missing: $POINTER"
  SOURCE_SELECTION_RUN_HOST=$(cat "$POINTER")
fi

[ -d "$SOURCE_SELECTION_RUN_HOST" ] || fail "selection run missing: $SOURCE_SELECTION_RUN_HOST"
[ -f "$SOURCE_SELECTION_RUN_HOST/ALL_STAGE2_GT_AUTOSELECT_V3_OK.txt" ] \
  || fail "selection run is not a successful completed target run: $SOURCE_SELECTION_RUN_HOST"
[ -f "$SOURCE_SELECTION_RUN_HOST/selection/selected_profile.env" ] \
  || fail "selected_profile.env missing"
[ -f "$SOURCE_SELECTION_RUN_HOST/selection/selected_profile.json" ] \
  || fail "selected_profile.json missing"
[ -f "$SOURCE_SELECTION_RUN_HOST/selection/selection_report.txt" ] \
  || fail "selection_report.txt missing"
[ -f "$SOURCE_SELECTION_RUN_HOST/runtime/selected_stage2_bundle.joblib" ] \
  || fail "selected Stage2 bundle missing"
[ -f "$SOURCE_SELECTION_RUN_HOST/runtime/selected_stage2_bundle.json" ] \
  || fail "selected Stage2 bundle metadata missing"

# shellcheck disable=SC1090
source "$SOURCE_SELECTION_RUN_HOST/selection/selected_profile.env"

[ "${SELECTION_SELECTED:-0}" = "1" ] || fail "selection profile was not accepted"
[ "${SELECTION_STATUS:-}" = "SELECTED_SAFE_GT_IMPROVEMENT" ] \
  || fail "unexpected selection status: ${SELECTION_STATUS:-missing}"

for value_name in LINE_CANDIDATE_THRESHOLD LINE_WEAK_THRESHOLD LINE_COMPETITION_RATIO LINE_REFINER_THRESHOLD; do
  value="${!value_name:-}"
  [ -n "$value" ] || fail "$value_name missing from selected_profile.env"
  is_number "$value" || fail "$value_name is not numeric: $value"
done

SELECTED_STAGE2_BUNDLE_HOST="$SOURCE_SELECTION_RUN_HOST/runtime/selected_stage2_bundle.joblib"
SELECTED_STAGE2_BUNDLE_C=$(host_to_container "$SELECTED_STAGE2_BUNDLE_HOST")
STAGE1_ROOT_C=$(host_to_container "$STAGE1_ROOT_HOST")
ORIGINAL_STAGE2_BUNDLE_C=$(host_to_container "$ORIGINAL_STAGE2_BUNDLE_HOST")

# Validate the selected bundle and scalar threshold in the exact container namespace.
docker run --rm -i \
  --mount "type=bind,source=$HOST_OUTPUTS,target=$CONTAINER_OUTPUTS,readonly" \
  "$IMAGE" \
  python - "$STAGE1_ROOT_C" "$ORIGINAL_STAGE2_BUNDLE_C" "$SELECTED_STAGE2_BUNDLE_C" "$LINE_REFINER_THRESHOLD" <<'PY'
import math
import sys
from pathlib import Path
import joblib

s1 = Path(sys.argv[1])
original_path = Path(sys.argv[2])
selected_path = Path(sys.argv[3])
expected_line_threshold = float(sys.argv[4])

if not s1.is_dir():
    raise SystemExit(f"Stage1 root missing in container: {s1}")
if not original_path.is_file():
    raise SystemExit(f"Original Stage2 bundle missing in container: {original_path}")
if not selected_path.is_file():
    raise SystemExit(f"Selected Stage2 bundle missing in container: {selected_path}")

original = joblib.load(original_path)
selected = joblib.load(selected_path)
for name, bundle in (("original", original), ("selected", selected)):
    for key in ("pole_model", "line_model", "pole_threshold", "line_threshold"):
        if key not in bundle:
            raise SystemExit(f"{name} bundle missing key {key}")

actual = float(selected["line_threshold"])
if not math.isclose(actual, expected_line_threshold, rel_tol=0.0, abs_tol=1e-12):
    raise SystemExit(
        f"selected bundle threshold mismatch: bundle={actual} env={expected_line_threshold}"
    )

print("CANONICAL_STAGE2_PATH_AND_BUNDLE_PREFLIGHT_OK")
print("stage1_root=", s1)
print("original_bundle=", original_path)
print("selected_bundle=", selected_path)
print("selected_line_refiner_threshold=", actual)
PY

mapfile -t S1_MANIFESTS < <(
  find "$STAGE1_ROOT_HOST" \
    -mindepth 2 \
    -maxdepth 2 \
    -type f \
    -name 'stage1_manifest.csv' \
    | sort
)
[ "${#S1_MANIFESTS[@]}" -gt 0 ] || fail "no Stage1 manifests found"

# Validate every Stage1 session mapping before creating the output root.
declare -A SEEN_GID=()
declare -A SEEN_SID=()
for manifest in "${S1_MANIFESTS[@]}"; do
  gid=$(extract_group_id "$manifest")
  [ -n "$gid" ] || fail "group_id missing in $manifest"
  sid=$(basename "$(dirname "$manifest")")
  expected_sid=$(printf '%s' "$gid" | sed -E 's/[^A-Za-z0-9_.-]+/__/g')
  [ "$sid" = "$expected_sid" ] \
    || fail "Stage1 directory/GID mismatch: actual=$sid expected=$expected_sid manifest=$manifest"
  [ -z "${SEEN_GID[$gid]:-}" ] || fail "duplicate group_id: $gid"
  [ -z "${SEEN_SID[$sid]:-}" ] || fail "duplicate SID: $sid"
  SEEN_GID[$gid]=1
  SEEN_SID[$sid]=1
  n=$(completed_count "$manifest" "$gid")
  [ "$n" -gt 0 ] || fail "no completed Stage1 rows for $gid"
done

COMMIT=$(git -C "$EXP_REPO" rev-parse HEAD)
COMMIT_SHORT=${COMMIT:0:12}
if [ -z "$RUN_STAMP" ]; then
  RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
fi
[[ "$RUN_STAMP" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || fail "RUN_STAMP must use YYYYMMDDTHHMMSSZ"

# Canonical, frozen layout: identical to the prior combined Stage2/Stage3 V2 run.
RUN_HOST="$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_stage23_quality/full_run_v2_${RUN_STAMP}"
RUN_C="$CONTAINER_OUTPUTS/poleline_voxel_run_session_groups/v4_stage23_quality/full_run_v2_${RUN_STAMP}"
STAGE2_HOST="$RUN_HOST/stage2"
STAGE2_C="$RUN_C/stage2"
STAGE3_HOST="$RUN_HOST/stage3"
LOG_HOST="$RUN_HOST/logs"
TIMING_HOST="$RUN_HOST/timing"
STATUS_HOST="$RUN_HOST/status"
SESSION_MAP="$RUN_HOST/session_map.tsv"

if [ "$PREFLIGHT_ONLY" = "1" ]; then
  echo "============================================================"
  echo "CANONICAL STAGE2 ALL-DATA PREFLIGHT OK"
  echo "============================================================"
  echo "repository=$EXP_REPO"
  echo "commit=$COMMIT"
  echo "source_selection_run=$SOURCE_SELECTION_RUN_HOST"
  echo "selected_profile=${SELECTED_PROFILE_NAME:-unknown}"
  echo "candidate=$LINE_CANDIDATE_THRESHOLD"
  echo "weak=$LINE_WEAK_THRESHOLD"
  echo "competition=$LINE_COMPETITION_RATIO"
  echo "refiner=$LINE_REFINER_THRESHOLD"
  echo "stage1_sessions=${#S1_MANIFESTS[@]}"
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
  "$LOG_HOST/stage2" \
  "$LOG_HOST/stage3" \
  "$TIMING_HOST/stage2" \
  "$STATUS_HOST"

printf '%s\n' "$RUN_HOST" > "$HOME/LATEST_V4_STAGE23_QUALITY_V2.txt"
printf '%s\n' "$RUN_HOST" > "$HOME/LATEST_V4_STAGE2_ALL_CANONICAL_RUN.txt"
: > "$SESSION_MAP"

cp "$SOURCE_SELECTION_RUN_HOST/selection/selected_profile.env" "$RUN_HOST/STAGE2_SELECTED_PROFILE.env"
cp "$SOURCE_SELECTION_RUN_HOST/selection/selected_profile.json" "$RUN_HOST/STAGE2_SELECTED_PROFILE.json"
cp "$SOURCE_SELECTION_RUN_HOST/selection/selection_report.txt" "$RUN_HOST/STAGE2_SELECTION_REPORT.txt"
cp "$SOURCE_SELECTION_RUN_HOST/runtime/selected_stage2_bundle.json" "$RUN_HOST/STAGE2_SELECTED_BUNDLE.json"

cat > "$RUN_HOST/RUN_INFO.txt" <<EOF
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
layout_contract=v4_stage23_quality/full_run_v2_<UTC>/stage2/<SID>
repository=$EXP_REPO
branch=$(git -C "$EXP_REPO" branch --show-current)
commit=$COMMIT
production_base=$PRODUCTION_TAG
source_selection_run=$SOURCE_SELECTION_RUN_HOST
selection_status=$SELECTION_STATUS
selected_profile=$SELECTED_PROFILE_NAME
stage1_source=$STAGE1_ROOT_HOST
stage1_rerun=false
original_stage2_bundle=$ORIGINAL_STAGE2_BUNDLE_HOST
selected_stage2_bundle=$SELECTED_STAGE2_BUNDLE_HOST
selected_stage2_bundle_sha256=$(sha256sum "$SELECTED_STAGE2_BUNDLE_HOST" | awk '{print $1}')
pole_candidate_threshold=$POLE_CANDIDATE_THRESHOLD
line_candidate_threshold=$LINE_CANDIDATE_THRESHOLD
line_weak_threshold=$LINE_WEAK_THRESHOLD
line_competition_ratio=$LINE_COMPETITION_RATIO
line_refiner_threshold=$LINE_REFINER_THRESHOLD
pole_min_voxels=$POLE_MIN_VOXELS
line_min_voxels=$LINE_MIN_VOXELS
edge_width_vox=$EDGE_WIDTH_VOX
max_slices=$MAX_SLICES
resume=$RESUME
stage2_scope=all_saved_stage1_sessions
stage3_ran=false
stage3_directory_reserved=$STAGE3_HOST
EOF

TOTAL=${#S1_MANIFESTS[@]}
INDEX=0
for manifest in "${S1_MANIFESTS[@]}"; do
  INDEX=$((INDEX + 1))
  GID=$(extract_group_id "$manifest")
  S1_SESSION_HOST=$(dirname "$manifest")
  SID=$(basename "$S1_SESSION_HOST")
  S1_SESSION_C=$(host_to_container "$S1_SESSION_HOST")
  S2_SESSION_HOST="$STAGE2_HOST/$SID"
  S2_SESSION_C="$STAGE2_C/$SID"
  T2_HOST="$TIMING_HOST/stage2/${SID}.csv"
  T2_C="$RUN_C/timing/stage2/${SID}.csv"
  LOG_FILE="$LOG_HOST/stage2/${SID}.log"
  STATUS_FILE="$STATUS_HOST/${SID}.stage2.ok"

  mkdir -p "$S2_SESSION_HOST"
  printf '%s\t%s\n' "$GID" "$SID" >> "$SESSION_MAP"

  echo "============================================================"
  echo "STAGE2 $INDEX/$TOTAL"
  echo "GID=$GID"
  echo "SID=$SID"
  echo "Stage1=$S1_SESSION_C"
  echo "Stage2=$S2_SESSION_C"
  echo "Timing=$T2_C"
  echo "candidate=$LINE_CANDIDATE_THRESHOLD weak=$LINE_WEAK_THRESHOLD competition=$LINE_COMPETITION_RATIO refiner=$LINE_REFINER_THRESHOLD"
  echo "============================================================"

  docker run --rm \
    --mount "type=bind,source=$EXP_REPO/v4,target=/workspace/v4,readonly" \
    --mount "type=bind,source=$EXP_REPO/ops/v4_full_dataset,target=/workspace/v4_full_ops,readonly" \
    --mount "type=bind,source=$HOST_OUTPUTS,target=$CONTAINER_OUTPUTS" \
    --workdir /workspace/v4 \
    -e PYTHONPATH=/workspace/v4 \
    -e PYTHONPYCACHEPREFIX=/tmp/pycache \
    -e MPLCONFIGDIR=/tmp/matplotlib \
    "$IMAGE" \
    python /workspace/v4_full_ops/profile_stage2_session.py \
      --stage1_dir "$S1_SESSION_C" \
      --output_dir "$S2_SESSION_C" \
      --session_filter "$GID" \
      --stage2_bundle "$SELECTED_STAGE2_BUNDLE_C" \
      --timing_csv "$T2_C" \
      --pole_candidate_threshold "$POLE_CANDIDATE_THRESHOLD" \
      --line_candidate_threshold "$LINE_CANDIDATE_THRESHOLD" \
      --line_weak_threshold "$LINE_WEAK_THRESHOLD" \
      --line_competition_ratio "$LINE_COMPETITION_RATIO" \
      --pole_min_voxels "$POLE_MIN_VOXELS" \
      --line_min_voxels "$LINE_MIN_VOXELS" \
      --edge_width_vox "$EDGE_WIDTH_VOX" \
      --resume "$RESUME" \
    2>&1 | tee "$LOG_FILE"

  [ -f "$S2_SESSION_HOST/STAGE2_COMPLETED.json" ] \
    || fail "Stage2 completion marker missing for $GID"
  [ -f "$S2_SESSION_HOST/inference_manifest.csv" ] \
    || fail "Stage2 inference manifest missing for $GID"
  [ -s "$T2_HOST" ] || fail "Stage2 timing CSV missing/empty for $GID"

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

  printf 'OK %s gid=%s rows=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$GID" "$OUTPUT_ROWS" \
    > "$STATUS_FILE"
done

EXPECTED_SESSIONS=$TOTAL
DONE_SESSIONS=$(find "$STATUS_HOST" -type f -name '*.stage2.ok' | wc -l | tr -d ' ')
MANIFEST_COUNT=$(find "$STAGE2_HOST" -mindepth 2 -maxdepth 2 -type f -name inference_manifest.csv | wc -l | tr -d ' ')
TIMING_COUNT=$(find "$TIMING_HOST/stage2" -maxdepth 1 -type f -name '*.csv' | wc -l | tr -d ' ')

[ "$DONE_SESSIONS" -eq "$EXPECTED_SESSIONS" ] \
  || fail "session completion mismatch: expected=$EXPECTED_SESSIONS done=$DONE_SESSIONS"
[ "$MANIFEST_COUNT" -eq "$EXPECTED_SESSIONS" ] \
  || fail "manifest count mismatch: expected=$EXPECTED_SESSIONS manifests=$MANIFEST_COUNT"
[ "$TIMING_COUNT" -eq "$EXPECTED_SESSIONS" ] \
  || fail "timing count mismatch: expected=$EXPECTED_SESSIONS timings=$TIMING_COUNT"

printf 'PHASE2_STAGE2_OK %s expected=%s completed=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$EXPECTED_SESSIONS" "$DONE_SESSIONS" \
  > "$RUN_HOST/PHASE2_STAGE2_OK.txt"
printf 'STAGE2_ONLY_COMPLETE %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$RUN_HOST/STAGE2_ONLY_COMPLETE.txt"
find "$RUN_HOST" -type f | sort > "$RUN_HOST/FILE_INVENTORY.txt"

PACKAGE_HOST="$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_stage23_quality/packages"
mkdir -p "$PACKAGE_HOST"
RUN_ID=$(basename "$RUN_HOST")
ARCHIVE="$PACKAGE_HOST/v4_stage2_recall_${RUN_ID}.tar.gz"

# Include the canonical directory skeleton. stage3/ and logs/stage3/ are empty
# reservations so a later Stage3 run can use this exact run root.
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
  logs \
  timing \
  status \
  RUN_INFO.txt \
  session_map.tsv \
  STAGE2_SELECTED_PROFILE.env \
  STAGE2_SELECTED_PROFILE.json \
  STAGE2_SELECTION_REPORT.txt \
  STAGE2_SELECTED_BUNDLE.json \
  PHASE2_STAGE2_OK.txt \
  STAGE2_ONLY_COMPLETE.txt \
  FILE_INVENTORY.txt

tar -tzf "$ARCHIVE" >/dev/null
sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
printf '%s\n' "$ARCHIVE" > "$HOME/LATEST_V4_STAGE2_RECALL_ARCHIVE.txt"
printf '%s\n' "$ARCHIVE" > "$HOME/LATEST_V4_STAGE2_ALL_CANONICAL_ARCHIVE.txt"

cat > "$RUN_HOST/STAGE2_PACKAGE_INFO.txt" <<EOF
archive=$ARCHIVE
archive_sha256=$(awk '{print $1}' "${ARCHIVE}.sha256")
packaged_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

echo "============================================================"
echo "V4 STAGE2 ALL-DATA CANONICAL RUN COMPLETE"
echo "============================================================"
echo "Run root:       $RUN_HOST"
echo "Stage2 root:    $STAGE2_HOST"
echo "Stage3 reserved:$STAGE3_HOST"
echo "Sessions:       $DONE_SESSIONS"
echo "Archive:        $ARCHIVE"
echo "Profile:        ${SELECTED_PROFILE_NAME:-unknown}"
echo "Thresholds:     candidate=$LINE_CANDIDATE_THRESHOLD weak=$LINE_WEAK_THRESHOLD competition=$LINE_COMPETITION_RATIO refiner=$LINE_REFINER_THRESHOLD"
echo "============================================================"
