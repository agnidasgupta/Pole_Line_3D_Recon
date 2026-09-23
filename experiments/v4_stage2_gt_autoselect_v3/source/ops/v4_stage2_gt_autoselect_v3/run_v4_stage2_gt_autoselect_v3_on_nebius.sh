#!/usr/bin/env bash
set -euo pipefail

# Offline GT-driven Stage-2 profile selection followed by a normal V4 Stage-2 run.
# Stage 1 and Stage 3 are never invoked by this script.

EXP_REPO="${EXP_REPO:-/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_gt_autoselect_v3}"
HOST_OUTPUTS="${HOST_OUTPUTS:-/workspace/voxel_poleline/outputs}"
CONTAINER_OUTPUTS="${CONTAINER_OUTPUTS:-/outputs}"
IMAGE="${IMAGE:-va-v4-realtime:torch241-cu121}"

BASELINE_RUN_HOST="${BASELINE_RUN_HOST:-$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_production/full_dataset_runs/d9977c39c443f5fa14f8/20260825T203403Z}"
STAGE1_ROOT_HOST="${STAGE1_ROOT_HOST:-$BASELINE_RUN_HOST/stage1}"
STAGE2_BUNDLE_HOST="${STAGE2_BUNDLE_HOST:-$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_realtime/stage2_refiner/local_refiner_bundle.joblib}"

TARGET_GID="${TARGET_GID:-VELASCO_CUT_CP/session1}"
RUN_SCOPE="${RUN_SCOPE:-target}"   # target or all
MAX_SLICES="${MAX_SLICES:-0}"
RESUME="${RESUME:-0}"
GUARDRAIL_SLICES_PER_SESSION="${GUARDRAIL_SLICES_PER_SESSION:-3}"
MAX_TARGET_SLICES="${MAX_TARGET_SLICES:-0}"

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

case "$RUN_SCOPE" in
  target|all) ;;
  *) fail "RUN_SCOPE must be target or all" ;;
esac

[ -d "$EXP_REPO/.git" ] || [ -f "$EXP_REPO/.git" ] || fail "experiment Git worktree missing: $EXP_REPO"
[ -f "$EXP_REPO/v4/run_v4_stage2.py" ] || fail "Stage2 entry point missing"
[ -f "$EXP_REPO/ops/v4_stage2_gt_autoselect_v3/select_v4_stage2_profile_with_gt.py" ] || fail "GT selector missing"
[ -f "$EXP_REPO/ops/v4_stage2_gt_autoselect_v3/prepare_selected_stage2_bundle.py" ] || fail "bundle preparer missing"
[ -d "$STAGE1_ROOT_HOST" ] || fail "saved Stage1 root missing: $STAGE1_ROOT_HOST"
[ -f "$STAGE2_BUNDLE_HOST" ] || fail "accepted Stage2 bundle missing: $STAGE2_BUNDLE_HOST"

if git -C "$EXP_REPO" grep -q 'recover_line_candidates_auto' -- v4 2>/dev/null; then
  fail "obsolete post-refiner recovery hook is present"
fi

if git -C "$EXP_REPO" rev-parse -q --verify 'v4.0.1-production-ops^{commit}' >/dev/null; then
  if ! git -C "$EXP_REPO" diff --quiet 'v4.0.1-production-ops' -- \
      v4/run_v4_stage2.py \
      v4/v4_realtime_pipeline.py \
      v4/v4_sparse_components.py \
      v4/v4_stage2_runtime.py \
      v4/v4_stage2_local.py; then
    fail "core Stage2 source differs from v4.0.1-production-ops"
  fi
else
  fail "production tag v4.0.1-production-ops is unavailable; fetch tags first"
fi

STAGE1_ROOT_C=$(host_to_container "$STAGE1_ROOT_HOST")
STAGE2_BUNDLE_C=$(host_to_container "$STAGE2_BUNDLE_HOST")
COMMIT=$(git -C "$EXP_REPO" rev-parse HEAD)
COMMIT_SHORT=${COMMIT:0:12}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_HOST="$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_stage2_gt_autoselect_v3/runs/$COMMIT_SHORT/$STAMP"
RUN_C="$CONTAINER_OUTPUTS/poleline_voxel_run_session_groups/v4_stage2_gt_autoselect_v3/runs/$COMMIT_SHORT/$STAMP"
SELECT_HOST="$RUN_HOST/selection"
SELECT_C="$RUN_C/selection"
STAGE2_HOST="$RUN_HOST/stage2"
LOG_HOST="$RUN_HOST/logs"
STATUS_HOST="$RUN_HOST/status"
RUNTIME_HOST="$RUN_HOST/runtime"
mkdir -p "$SELECT_HOST" "$STAGE2_HOST" "$LOG_HOST" "$STATUS_HOST" "$RUNTIME_HOST"
printf '%s\n' "$RUN_HOST" > "$HOME/LATEST_V4_STAGE2_GT_AUTOSELECT_V3_RUN.txt"

cat > "$RUN_HOST/RUN_INFO.txt" <<EOF
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
repository=$EXP_REPO
branch=$(git -C "$EXP_REPO" branch --show-current)
commit=$COMMIT
production_base=v4.0.1-production-ops
target_session=$TARGET_GID
run_scope=$RUN_SCOPE
stage1_root=$STAGE1_ROOT_HOST
stage1_rerun=false
stage2_bundle=$STAGE2_BUNDLE_HOST
stage2_bundle_sha256=$(sha256sum "$STAGE2_BUNDLE_HOST" | awk '{print $1}')
ground_truth_source=saved_stage1_npz_raw_labels_label_2
ground_truth_use=offline_profile_selection_only
runtime_gt_features=false
guardrail_slices_per_session=$GUARDRAIL_SLICES_PER_SESSION
max_target_slices=$MAX_TARGET_SLICES
max_output_slices=$MAX_SLICES
stage3_ran=false
EOF

# Validate input and bundle in the exact container namespace.
docker run --rm -i \
  --mount "type=bind,source=$EXP_REPO/v4,target=/workspace/v4,readonly" \
  --mount "type=bind,source=$EXP_REPO/ops/v4_stage2_gt_autoselect_v3,target=/workspace/quality,readonly" \
  --mount "type=bind,source=$HOST_OUTPUTS,target=$CONTAINER_OUTPUTS" \
  --workdir /workspace/v4 \
  -e PYTHONPATH=/workspace/v4 \
  "$IMAGE" \
  python - "$STAGE1_ROOT_C" "$STAGE2_BUNDLE_C" <<'PY'
import sys
from pathlib import Path
import joblib
s1=Path(sys.argv[1]); bundle=Path(sys.argv[2])
if not s1.is_dir(): raise SystemExit(f"Stage1 root missing: {s1}")
if not bundle.is_file(): raise SystemExit(f"Stage2 bundle missing: {bundle}")
b=joblib.load(bundle)
for k in ["pole_model","line_model","pole_threshold","line_threshold"]:
    if k not in b: raise SystemExit(f"bundle missing {k}")
print("GT_AUTOSELECT_PATH_PREFLIGHT_OK")
print("stage1_root=", s1)
print("stage2_bundle=", bundle)
print("bundle_line_threshold=", b["line_threshold"])
PY

# Static/self-test of selection logic.
docker run --rm \
  --mount "type=bind,source=$EXP_REPO/v4,target=/workspace/v4,readonly" \
  --mount "type=bind,source=$EXP_REPO/ops/v4_stage2_gt_autoselect_v3,target=/workspace/quality,readonly" \
  --workdir /workspace/v4 \
  -e PYTHONPATH=/workspace/v4 \
  "$IMAGE" \
  python /workspace/quality/select_v4_stage2_profile_with_gt.py --self_test

# Offline GT-driven selection. This does not write Stage2 outputs.
docker run --rm \
  --mount "type=bind,source=$EXP_REPO/v4,target=/workspace/v4,readonly" \
  --mount "type=bind,source=$EXP_REPO/ops/v4_stage2_gt_autoselect_v3,target=/workspace/quality,readonly" \
  --mount "type=bind,source=$HOST_OUTPUTS,target=$CONTAINER_OUTPUTS" \
  --workdir /workspace/v4 \
  -e PYTHONPATH=/workspace/v4 \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPYCACHEPREFIX=/tmp/pycache \
  "$IMAGE" \
  python /workspace/quality/select_v4_stage2_profile_with_gt.py \
    --stage1_root "$STAGE1_ROOT_C" \
    --stage2_bundle "$STAGE2_BUNDLE_C" \
    --target_session "$TARGET_GID" \
    --output_dir "$SELECT_C" \
    --pole_candidate_threshold "$POLE_CANDIDATE_THRESHOLD" \
    --pole_min_voxels "$POLE_MIN_VOXELS" \
    --line_min_voxels "$LINE_MIN_VOXELS" \
    --edge_width_vox "$EDGE_WIDTH_VOX" \
    --max_target_slices "$MAX_TARGET_SLICES" \
    --guardrail_slices_per_session "$GUARDRAIL_SLICES_PER_SESSION" \
  2>&1 | tee "$LOG_HOST/selection.log"

[ -f "$SELECT_HOST/selected_profile.env" ] || fail "selector did not write selected_profile.env"
# shellcheck disable=SC1090
source "$SELECT_HOST/selected_profile.env"

cat "$SELECT_HOST/selection_report.txt"

if [ "${SELECTION_SELECTED:-0}" != "1" ]; then
  printf 'NO_SAFE_GT_IMPROVEMENT %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_HOST/NO_SAFE_GT_IMPROVEMENT.txt"
  PACKAGE_ROOT="$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_stage2_gt_autoselect_v3/packages"
  mkdir -p "$PACKAGE_ROOT"
  ARCHIVE="$PACKAGE_ROOT/v4_stage2_gt_autoselect_v3_diagnostics_${COMMIT_SHORT}_${STAMP}.tar.gz"
  tar -czf "$ARCHIVE" -C "$RUN_HOST" selection logs RUN_INFO.txt NO_SAFE_GT_IMPROVEMENT.txt
  tar -tzf "$ARCHIVE" >/dev/null
  sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
  printf '%s\n' "$ARCHIVE" > "$HOME/LATEST_V4_STAGE2_GT_AUTOSELECT_V3_ARCHIVE.txt"
  echo "NO_SAFE_GT_IMPROVEMENT; diagnostics packaged at $ARCHIVE"
  exit 3
fi

# Build an experiment-only bundle with the selected scalar refiner threshold.
SELECTED_BUNDLE_HOST="$RUNTIME_HOST/selected_stage2_bundle.joblib"
SELECTED_BUNDLE_C="$RUN_C/runtime/selected_stage2_bundle.joblib"
SELECTED_BUNDLE_META_HOST="$RUNTIME_HOST/selected_stage2_bundle.json"
SELECTED_BUNDLE_META_C="$RUN_C/runtime/selected_stage2_bundle.json"

docker run --rm \
  --mount "type=bind,source=$EXP_REPO/v4,target=/workspace/v4,readonly" \
  --mount "type=bind,source=$EXP_REPO/ops/v4_stage2_gt_autoselect_v3,target=/workspace/quality,readonly" \
  --mount "type=bind,source=$HOST_OUTPUTS,target=$CONTAINER_OUTPUTS" \
  --workdir /workspace/v4 \
  -e PYTHONPATH=/workspace/v4 \
  "$IMAGE" \
  python /workspace/quality/prepare_selected_stage2_bundle.py \
    --input_bundle "$STAGE2_BUNDLE_C" \
    --selection_json "$SELECT_C/selected_profile.json" \
    --output_bundle "$SELECTED_BUNDLE_C" \
    --output_metadata "$SELECTED_BUNDLE_META_C"

[ -f "$SELECTED_BUNDLE_HOST" ] || fail "selected runtime bundle missing"

mapfile -t ALL_MANIFESTS < <(
  find "$STAGE1_ROOT_HOST" -mindepth 2 -maxdepth 2 -type f -name stage1_manifest.csv | sort
)
[ "${#ALL_MANIFESTS[@]}" -gt 0 ] || fail "no Stage1 manifests found"
SELECTED_MANIFESTS=()
for manifest in "${ALL_MANIFESTS[@]}"; do
  gid=$(extract_group_id "$manifest")
  [ -n "$gid" ] || fail "group_id missing in $manifest"
  if [ "$RUN_SCOPE" = "all" ] || [ "$gid" = "$TARGET_GID" ]; then
    SELECTED_MANIFESTS+=("$manifest")
  fi
done
[ "${#SELECTED_MANIFESTS[@]}" -gt 0 ] || fail "no manifests selected for RUN_SCOPE=$RUN_SCOPE"

SESSION_MAP="$RUN_HOST/session_map.tsv"
printf 'group_id\tsid\tstage1_host\tstage2_host\n' > "$SESSION_MAP"

for manifest in "${SELECTED_MANIFESTS[@]}"; do
  gid=$(extract_group_id "$manifest")
  s1_host=$(dirname "$manifest")
  sid=$(basename "$s1_host")
  expected_sid=$(printf '%s' "$gid" | sed -E 's/[^A-Za-z0-9_.-]+/__/g')
  [ "$sid" = "$expected_sid" ] || fail "SID mismatch actual=$sid expected=$expected_sid"
  s1_c=$(host_to_container "$s1_host")
  out_host="$STAGE2_HOST/$sid"
  out_c="$RUN_C/stage2/$sid"
  mkdir -p "$out_host"
  printf '%s\t%s\t%s\t%s\n' "$gid" "$sid" "$s1_host" "$out_host" >> "$SESSION_MAP"
  echo "============================================================"
  echo "Selected Stage2 gid=$gid"
  echo "candidate=$LINE_CANDIDATE_THRESHOLD weak=$LINE_WEAK_THRESHOLD competition=$LINE_COMPETITION_RATIO refiner=$LINE_REFINER_THRESHOLD"
  echo "Stage1: $s1_c"
  echo "Stage2: $out_c"
  echo "============================================================"
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
      --stage2_bundle "$SELECTED_BUNDLE_C" \
      --pole_candidate_threshold "$POLE_CANDIDATE_THRESHOLD" \
      --line_candidate_threshold "$LINE_CANDIDATE_THRESHOLD" \
      --line_weak_threshold "$LINE_WEAK_THRESHOLD" \
      --line_competition_ratio "$LINE_COMPETITION_RATIO" \
      --pole_min_voxels "$POLE_MIN_VOXELS" \
      --line_min_voxels "$LINE_MIN_VOXELS" \
      --edge_width_vox "$EDGE_WIDTH_VOX" \
      --resume "$RESUME" \
      --max_slices "$MAX_SLICES" \
    2>&1 | tee "$LOG_HOST/stage2__${sid}.log"
  [ -f "$out_host/inference_manifest.csv" ] || fail "Stage2 manifest missing for $gid"
  rows=$(awk 'END {print NR-1}' "$out_host/inference_manifest.csv")
  [ "$rows" -gt 0 ] || fail "Stage2 manifest has no data rows for $gid"
  if grep -qiE 'Traceback|IsADirectoryError|FileNotFoundError|RuntimeError|Exception:' "$LOG_HOST/stage2__${sid}.log"; then
    fail "failure found in Stage2 log for $gid"
  fi
  printf 'OK %s gid=%s rows=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$gid" "$rows" > "$STATUS_HOST/${sid}.ok"
done

EXPECTED=${#SELECTED_MANIFESTS[@]}
DONE=$(find "$STATUS_HOST" -type f -name '*.ok' | wc -l | tr -d ' ')
[ "$DONE" -eq "$EXPECTED" ] || fail "completion mismatch expected=$EXPECTED done=$DONE"
printf 'PHASE2_STAGE2_OK %s expected=%s completed=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$EXPECTED" "$DONE" > "$RUN_HOST/PHASE2_STAGE2_OK.txt"
printf 'ALL_STAGE2_GT_AUTOSELECT_V3_OK %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_HOST/ALL_STAGE2_GT_AUTOSELECT_V3_OK.txt"
find "$RUN_HOST" -type f | sort > "$RUN_HOST/FILE_INVENTORY.txt"

PACKAGE_ROOT="$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_stage2_gt_autoselect_v3/packages"
mkdir -p "$PACKAGE_ROOT"
ARCHIVE="$PACKAGE_ROOT/v4_stage2_gt_autoselect_v3_${RUN_SCOPE}_${COMMIT_SHORT}_${STAMP}.tar.gz"
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
  stage2 selection logs status RUN_INFO.txt session_map.tsv PHASE2_STAGE2_OK.txt ALL_STAGE2_GT_AUTOSELECT_V3_OK.txt FILE_INVENTORY.txt runtime/selected_stage2_bundle.json

tar -tzf "$ARCHIVE" >/dev/null
sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"
printf '%s\n' "$ARCHIVE" > "$HOME/LATEST_V4_STAGE2_GT_AUTOSELECT_V3_ARCHIVE.txt"
echo "============================================================"
echo "V4 STAGE2 GT AUTOSELECT V3 COMPLETE"
echo "Run root: $RUN_HOST"
echo "Archive:  $ARCHIVE"
echo "Selected: $SELECTED_PROFILE_NAME"
echo "Thresholds: candidate=$LINE_CANDIDATE_THRESHOLD weak=$LINE_WEAK_THRESHOLD competition=$LINE_COMPETITION_RATIO refiner=$LINE_REFINER_THRESHOLD"
echo "============================================================"
