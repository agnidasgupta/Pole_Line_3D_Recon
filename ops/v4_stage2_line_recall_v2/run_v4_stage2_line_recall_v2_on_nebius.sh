#!/usr/bin/env bash
set -euo pipefail

# Stage-2-only experiment. Reuses accepted V4 Stage-1 artifacts and the accepted
# Stage-2 refiner bundle. It does not run Stage 1 or Stage 3.

EXP_REPO="${EXP_REPO:-/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_line_recall_v2}"
HOST_OUTPUTS="${HOST_OUTPUTS:-/workspace/voxel_poleline/outputs}"
CONTAINER_OUTPUTS="${CONTAINER_OUTPUTS:-/outputs}"
IMAGE="${IMAGE:-va-v4-realtime:torch241-cu121}"

BASELINE_RUN_HOST="${BASELINE_RUN_HOST:-$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_production/full_dataset_runs/d9977c39c443f5fa14f8/20260825T203403Z}"
STAGE1_ROOT_HOST="${STAGE1_ROOT_HOST:-$BASELINE_RUN_HOST/stage1}"
STAGE2_BUNDLE_HOST="${STAGE2_BUNDLE_HOST:-$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_realtime/stage2_refiner/local_refiner_bundle.joblib}"

# velasco_sweep: run three candidate settings on VELASCO_CUT_CP/session1.
# all: run every saved Stage-1 session with SELECTED_PROFILE.
RUN_SCOPE="${RUN_SCOPE:-velasco_sweep}"
TARGET_GID="${TARGET_GID:-VELASCO_CUT_CP/session1}"
SELECTED_PROFILE="${SELECTED_PROFILE:-recall_mid}"
MAX_SLICES="${MAX_SLICES:-0}"
RESUME="${RESUME:-0}"

# Production settings kept unchanged unless explicitly listed per profile.
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
      fail "path is outside HOST_OUTPUTS: $p"
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

profile_values() {
  case "$1" in
    baseline)
      printf '%s\t%s\t%s\n' '0.08' '0.04' '0.55'
      ;;
    recall_mid)
      printf '%s\t%s\t%s\n' '0.04' '0.01' '0.35'
      ;;
    recall_high)
      printf '%s\t%s\t%s\n' '0.025' '0.005' '0.20'
      ;;
    *)
      fail "unknown profile '$1'; expected baseline, recall_mid, or recall_high"
      ;;
  esac
}

[ -d "$EXP_REPO/.git" ] || fail "experiment repository missing: $EXP_REPO"
[ -d "$EXP_REPO/v4" ] || fail "v4 directory missing: $EXP_REPO/v4"
[ -f "$EXP_REPO/v4/run_v4_stage2.py" ] || fail "run_v4_stage2.py missing"
[ -f "$EXP_REPO/v4/v4_stage2_runtime.py" ] || fail "v4_stage2_runtime.py missing"
[ -d "$STAGE1_ROOT_HOST" ] || fail "saved Stage-1 root missing: $STAGE1_ROOT_HOST"
[ -f "$STAGE2_BUNDLE_HOST" ] || fail "Stage-2 bundle is not a file: $STAGE2_BUNDLE_HOST"

# This experiment must use the production Stage-2 implementation, not the
# obsolete post-refiner recovery hook.
if grep -q 'recover_line_candidates_auto' "$EXP_REPO/v4/v4_stage2_runtime.py"; then
  fail "obsolete recover_line_candidates_auto hook remains in v4_stage2_runtime.py"
fi

# The experiment branch is expected to leave core Stage-2 source identical to
# the accepted V4 tag and change only operations/runbooks.
if git -C "$EXP_REPO" rev-parse -q --verify 'v4.0.1-production-ops^{commit}' >/dev/null; then
  if ! git -C "$EXP_REPO" diff --quiet 'v4.0.1-production-ops' -- \
      v4/v4_stage2_runtime.py \
      v4/v4_realtime_pipeline.py \
      v4/v4_stage2_local.py \
      v4/run_v4_stage2.py; then
    fail "core Stage-2 source differs from v4.0.1-production-ops"
  fi
else
  fail "production tag v4.0.1-production-ops is not available in the clone"
fi

STAGE2_BUNDLE_C=$(host_to_container "$STAGE2_BUNDLE_HOST")

mapfile -t ALL_MANIFESTS < <(
  find "$STAGE1_ROOT_HOST" \
    -mindepth 2 \
    -maxdepth 2 \
    -type f \
    -name 'stage1_manifest.csv' \
    | sort
)
[ "${#ALL_MANIFESTS[@]}" -gt 0 ] || fail "no stage1_manifest.csv files found under $STAGE1_ROOT_HOST"

SELECTED_MANIFESTS=()
for manifest in "${ALL_MANIFESTS[@]}"; do
  gid=$(extract_group_id "$manifest")
  [ -n "$gid" ] || fail "group_id missing in $manifest"
  case "$RUN_SCOPE" in
    velasco_sweep)
      [ "$gid" = "$TARGET_GID" ] && SELECTED_MANIFESTS+=("$manifest")
      ;;
    all)
      SELECTED_MANIFESTS+=("$manifest")
      ;;
    *)
      fail "RUN_SCOPE must be velasco_sweep or all"
      ;;
  esac
done

[ "${#SELECTED_MANIFESTS[@]}" -gt 0 ] || fail "no Stage-1 session matched RUN_SCOPE=$RUN_SCOPE TARGET_GID=$TARGET_GID"

COMMIT=$(git -C "$EXP_REPO" rev-parse HEAD)
COMMIT_SHORT=${COMMIT:0:12}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

case "$RUN_SCOPE" in
  velasco_sweep)
    RUN_CLASS="sweeps"
    PROFILES=(baseline recall_mid recall_high)
    ;;
  all)
    RUN_CLASS="full_runs"
    PROFILES=("$SELECTED_PROFILE")
    ;;
esac

RUN_HOST="$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_stage2_line_recall_v2/$RUN_CLASS/${COMMIT_SHORT}/${STAMP}"
RUN_C="$CONTAINER_OUTPUTS/poleline_voxel_run_session_groups/v4_stage2_line_recall_v2/$RUN_CLASS/${COMMIT_SHORT}/${STAMP}"
LOG_HOST="$RUN_HOST/logs"
STATUS_HOST="$RUN_HOST/status"
SESSION_MAP="$RUN_HOST/session_map.tsv"

mkdir -p "$RUN_HOST" "$LOG_HOST" "$STATUS_HOST"
printf '%s\n' "$RUN_HOST" > "$HOME/LATEST_V4_STAGE2_LINE_RECALL_V2_RUN.txt"
: > "$SESSION_MAP"

cat > "$RUN_HOST/RUN_INFO.txt" <<EOF
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
repository=$EXP_REPO
branch=$(git -C "$EXP_REPO" branch --show-current)
commit=$COMMIT
production_base=v4.0.1-production-ops
run_scope=$RUN_SCOPE
target_gid=$TARGET_GID
selected_profile=$SELECTED_PROFILE
stage1_root=$STAGE1_ROOT_HOST
stage1_rerun=false
stage2_bundle=$STAGE2_BUNDLE_HOST
stage2_bundle_sha256=$(sha256sum "$STAGE2_BUNDLE_HOST" | awk '{print $1}')
pole_candidate_threshold=$POLE_CANDIDATE_THRESHOLD
pole_min_voxels=$POLE_MIN_VOXELS
line_min_voxels=$LINE_MIN_VOXELS
edge_width_vox=$EDGE_WIDTH_VOX
max_slices=$MAX_SLICES
resume=$RESUME
stage3_ran=false
EOF

# Validate every fixed path inside the exact Docker mount before processing.
docker run --rm -i \
  --mount "type=bind,source=$EXP_REPO/v4,target=/workspace/v4,readonly" \
  --mount "type=bind,source=$HOST_OUTPUTS,target=$CONTAINER_OUTPUTS" \
  --workdir /workspace/v4 \
  -e PYTHONPATH=/workspace/v4 \
  "$IMAGE" \
  python - "$STAGE2_BUNDLE_C" <<'PY'
import sys
from pathlib import Path
import joblib
p = Path(sys.argv[1])
if not p.is_file():
    raise SystemExit(f"bundle is not a file inside container: {p}")
b = joblib.load(p)
print("STAGE2_BUNDLE_CONTAINER_LOAD_OK")
print("bundle_type=", type(b).__name__)
if isinstance(b, dict):
    print("bundle_keys=", sorted(b))
PY

run_one() {
  local profile="$1"
  local manifest="$2"
  local gid sid s1_host s1_c out_host out_c log_file
  local line_candidate line_weak competition

  gid=$(extract_group_id "$manifest")
  s1_host=$(dirname "$manifest")
  sid=$(basename "$s1_host")
  s1_c=$(host_to_container "$s1_host")

  expected_sid=$(printf '%s' "$gid" | sed -E 's/[^A-Za-z0-9_.-]+/__/g')
  [ "$sid" = "$expected_sid" ] || fail "SID mismatch: actual=$sid expected=$expected_sid gid=$gid"

  IFS=$'\t' read -r line_candidate line_weak competition < <(profile_values "$profile")

  out_host="$RUN_HOST/profiles/$profile/stage2/$sid"
  out_c="$RUN_C/profiles/$profile/stage2/$sid"
  log_file="$LOG_HOST/${profile}__${sid}.log"
  mkdir -p "$out_host"

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$profile" "$gid" "$sid" "$line_candidate" "$line_weak" "$competition" \
    >> "$SESSION_MAP"

  echo "============================================================"
  echo "STAGE2 profile=$profile gid=$gid"
  echo "Stage1 host:      $s1_host"
  echo "Stage1 container: $s1_c"
  echo "Stage2 host:      $out_host"
  echo "Stage2 container: $out_c"
  echo "Bundle container: $STAGE2_BUNDLE_C"
  echo "Thresholds: candidate=$line_candidate weak=$line_weak competition=$competition"
  echo "============================================================"

  # Per-session path preflight inside the same mount used by Stage 2.
  docker run --rm \
    --mount "type=bind,source=$HOST_OUTPUTS,target=$CONTAINER_OUTPUTS" \
    "$IMAGE" \
    bash -lc "
      set -euo pipefail
      test -f '$s1_c/stage1_manifest.csv'
      test -f '$STAGE2_BUNDLE_C'
      test -d '$out_c'
      echo SESSION_PATH_PREFLIGHT_OK
    "

  docker run --rm \
    --mount "type=bind,source=$EXP_REPO/v4,target=/workspace/v4,readonly" \
    --mount "type=bind,source=$HOST_OUTPUTS,target=$CONTAINER_OUTPUTS" \
    --workdir /workspace/v4 \
    -e PYTHONPATH=/workspace/v4 \
    -e PYTHONPYCACHEPREFIX=/tmp/pycache \
    -e MPLCONFIGDIR=/tmp/matplotlib \
    "$IMAGE" \
    python /workspace/v4/run_v4_stage2.py \
      --stage1_dir "$s1_c" \
      --output_dir "$out_c" \
      --session_filter "$gid" \
      --stage2_bundle "$STAGE2_BUNDLE_C" \
      --pole_candidate_threshold "$POLE_CANDIDATE_THRESHOLD" \
      --line_candidate_threshold "$line_candidate" \
      --line_weak_threshold "$line_weak" \
      --line_competition_ratio "$competition" \
      --pole_min_voxels "$POLE_MIN_VOXELS" \
      --line_min_voxels "$LINE_MIN_VOXELS" \
      --edge_width_vox "$EDGE_WIDTH_VOX" \
      --resume "$RESUME" \
      --max_slices "$MAX_SLICES" \
    2>&1 | tee "$log_file"

  [ -f "$out_host/inference_manifest.csv" ] || fail "Stage-2 manifest missing: $out_host/inference_manifest.csv"
  rows=$(awk 'END {print NR-1}' "$out_host/inference_manifest.csv")
  [ "$rows" -gt 0 ] || fail "Stage-2 manifest has no data rows: $out_host/inference_manifest.csv"

  if grep -qiE 'Traceback|IsADirectoryError|FileNotFoundError|RuntimeError|Exception:' "$log_file"; then
    fail "error-like failure found in $log_file"
  fi

  printf 'OK %s profile=%s gid=%s rows=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$profile" "$gid" "$rows" \
    > "$STATUS_HOST/${profile}__${sid}.ok"
}

for profile in "${PROFILES[@]}"; do
  for manifest in "${SELECTED_MANIFESTS[@]}"; do
    run_one "$profile" "$manifest"
  done
done

EXPECTED=$((${#PROFILES[@]} * ${#SELECTED_MANIFESTS[@]}))
DONE=$(find "$STATUS_HOST" -type f -name '*.ok' | wc -l | tr -d ' ')
[ "$DONE" -eq "$EXPECTED" ] || fail "completion count mismatch: expected=$EXPECTED done=$DONE"

printf 'PHASE2_STAGE2_OK %s expected=%s completed=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$EXPECTED" "$DONE" \
  > "$RUN_HOST/PHASE2_STAGE2_OK.txt"
printf 'ALL_STAGE2_V2_OK %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_HOST/ALL_STAGE2_V2_OK.txt"
find "$RUN_HOST" -type f | sort > "$RUN_HOST/FILE_INVENTORY.txt"

PACKAGE_ROOT="$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_stage2_line_recall_v2/packages"
mkdir -p "$PACKAGE_ROOT"
ARCHIVE="$PACKAGE_ROOT/v4_stage2_line_recall_v2_${RUN_SCOPE}_${COMMIT_SHORT}_${STAMP}.tar.gz"

tar -czf "$ARCHIVE" \
  --exclude='*.npz' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.ckpt' \
  --exclude='*.onnx' \
  --exclude='*.engine' \
  --exclude='*.safetensors' \
  --exclude='*.joblib' \
  -C "$(dirname "$RUN_HOST")" \
  "$(basename "$RUN_HOST")"

tar -tzf "$ARCHIVE" >/dev/null
sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
printf '%s\n' "$ARCHIVE" > "$HOME/LATEST_V4_STAGE2_LINE_RECALL_V2_ARCHIVE.txt"
echo "============================================================"
echo "V4 STAGE2 LINE-RECALL V2 COMPLETE"
echo "Run root: $RUN_HOST"
echo "Archive:  $ARCHIVE"
echo "Profiles: ${PROFILES[*]}"
echo "Sessions: ${#SELECTED_MANIFESTS[@]}"
echo "============================================================"
