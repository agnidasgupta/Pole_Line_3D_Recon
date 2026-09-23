#!/usr/bin/env bash
set -euo pipefail

EXP_REPO="/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage23_quality"
BASELINE_RUN_HOST="/workspace/voxel_poleline/outputs/poleline_voxel_run_session_groups/v4_production/full_dataset_runs/d9977c39c443f5fa14f8/20260825T203403Z"
BASELINE_RUN_C="/outputs/poleline_voxel_run_session_groups/v4_production/full_dataset_runs/d9977c39c443f5fa14f8/20260825T203403Z"
STAGE1_HOST="$BASELINE_RUN_HOST/stage1"
IMAGE="va-v4-realtime:torch241-cu121"
HOST_OUTPUTS="/workspace/voxel_poleline/outputs"
CONTAINER_OUTPUTS="/outputs"

# Quality-first Stage-2 line candidate settings.
LINE_CANDIDATE_THRESHOLD="${LINE_CANDIDATE_THRESHOLD:-0.05}"
LINE_WEAK_THRESHOLD="${LINE_WEAK_THRESHOLD:-0.015}"
LINE_COMPETITION_RATIO="${LINE_COMPETITION_RATIO:-0.40}"

# Keep accepted pole and component-size settings unchanged.
POLE_CANDIDATE_THRESHOLD="${POLE_CANDIDATE_THRESHOLD:-0.15}"
POLE_MIN_VOXELS="${POLE_MIN_VOXELS:-4}"
LINE_MIN_VOXELS="${LINE_MIN_VOXELS:-3}"
EDGE_WIDTH_VOX="${EDGE_WIDTH_VOX:-10}"

if [ ! -d "$EXP_REPO/.git" ]; then
  echo "ERROR: experiment repository not found: $EXP_REPO" >&2
  exit 2
fi
if [ ! -d "$STAGE1_HOST" ]; then
  echo "ERROR: saved Stage1 root not found: $STAGE1_HOST" >&2
  exit 3
fi

# Use the accepted V4 Stage-2 refiner bundle.
DEFAULT_STAGE2_BUNDLE="$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_realtime/stage2_refiner/local_refiner_bundle.joblib"
STAGE2_BUNDLE_HOST="${STAGE2_BUNDLE_HOST:-$DEFAULT_STAGE2_BUNDLE}"
if [ ! -f "$STAGE2_BUNDLE_HOST" ]; then
  echo "ERROR: accepted Stage2 local_refiner_bundle.joblib not found at:" >&2
  echo "  $STAGE2_BUNDLE_HOST" >&2
  echo "Set STAGE2_BUNDLE_HOST explicitly only if the accepted bundle is stored elsewhere." >&2
  exit 4
fi
case "$STAGE2_BUNDLE_HOST" in
  "$HOST_OUTPUTS"/*) STAGE2_BUNDLE_C="$CONTAINER_OUTPUTS${STAGE2_BUNDLE_HOST#$HOST_OUTPUTS}" ;;
  *) echo "ERROR: Stage2 bundle is outside mounted output root: $STAGE2_BUNDLE_HOST" >&2; exit 5 ;;
esac

# Guard against the previously incorrect refiner-layer hook.
if grep -q 'recover_line_candidates_auto' "$EXP_REPO/v4/v4_stage2_runtime.py"; then
  echo "ERROR: obsolete recover_line_candidates_auto hook is still present in v4_stage2_runtime.py" >&2
  echo "Revert v4_stage2_runtime.py to v4.0.1-production-ops before running." >&2
  exit 6
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_HOST="$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_stage23_quality/full_run_v2_${STAMP}"
RUN_C="$CONTAINER_OUTPUTS/poleline_voxel_run_session_groups/v4_stage23_quality/full_run_v2_${STAMP}"
STAGE2_HOST="$RUN_HOST/stage2"
STAGE2_C="$RUN_C/stage2"
STAGE3_HOST="$RUN_HOST/stage3"
STAGE3_C="$RUN_C/stage3"
LOG_HOST="$RUN_HOST/logs"
TIMING_HOST="$RUN_HOST/timing"
STATUS_HOST="$RUN_HOST/status"
SESSION_MAP="$RUN_HOST/session_map.tsv"

mkdir -p "$STAGE2_HOST" "$STAGE3_HOST" "$LOG_HOST/stage2" "$LOG_HOST/stage3" "$TIMING_HOST/stage2" "$STATUS_HOST"
: > "$SESSION_MAP"
printf '%s\n' "$RUN_HOST" > "$HOME/LATEST_V4_STAGE23_QUALITY_V2.txt"

cat > "$RUN_HOST/RUN_INFO.txt" <<EOF
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
branch=$(git -C "$EXP_REPO" branch --show-current)
commit=$(git -C "$EXP_REPO" rev-parse HEAD)
stage1_source=$STAGE1_HOST
stage1_rerun=false
stage2_bundle=$STAGE2_BUNDLE_HOST
pole_candidate_threshold=$POLE_CANDIDATE_THRESHOLD
line_candidate_threshold=$LINE_CANDIDATE_THRESHOLD
line_weak_threshold=$LINE_WEAK_THRESHOLD
line_competition_ratio=$LINE_COMPETITION_RATIO
pole_min_voxels=$POLE_MIN_VOXELS
line_min_voxels=$LINE_MIN_VOXELS
edge_width_vox=$EDGE_WIDTH_VOX
stage3_execution=fresh_process_full_legal_window
max_sequence_gap=9
slice_length_ft=50
max_span_length_ft=450
EOF

mapfile -t S1_MANIFESTS < <(find "$STAGE1_HOST" -type f -name 'stage1_manifest.csv' | sort)
if [ "${#S1_MANIFESTS[@]}" -eq 0 ]; then
  echo "ERROR: no stage1_manifest.csv files found below $STAGE1_HOST" >&2
  exit 10
fi

echo "Stage1 session manifests found: ${#S1_MANIFESTS[@]}"
echo "Stage2 bundle: $STAGE2_BUNDLE_HOST"
echo "Output root: $RUN_HOST"

extract_group_id() {
  local manifest="$1"
  awk -F',' '
    NR==1 {
      for (i=1; i<=NF; i++) {
        h=$i; gsub(/^"|"$/, "", h)
        if (h=="group_id") c=i
      }
      next
    }
    NR>1 && c {
      v=$c; gsub(/^"|"$/, "", v)
      if (v!="") { print v; exit }
    }
  ' "$manifest"
}

host_to_container_output_path() {
  local p="$1"
  case "$p" in
    "$HOST_OUTPUTS"/*) printf '%s%s\n' "$CONTAINER_OUTPUTS" "${p#$HOST_OUTPUTS}" ;;
    *) return 1 ;;
  esac
}

###############################################################################
# STAGE 2 - all sessions from saved Stage 1
###############################################################################

echo "============================================================"
echo "STAGE 2 QUALITY RUN"
echo "============================================================"

idx=0
total=${#S1_MANIFESTS[@]}
for manifest in "${S1_MANIFESTS[@]}"; do
  idx=$((idx + 1))
  GID=$(extract_group_id "$manifest")
  if [ -z "$GID" ]; then
    echo "ERROR: group_id not found in $manifest" >&2
    exit 11
  fi
  SID=$(printf '%s' "$GID" | sed -E 's/[^A-Za-z0-9_.-]+/__/g')
  S1_SESSION_HOST=$(dirname "$manifest")
  S1_SESSION_C=$(host_to_container_output_path "$S1_SESSION_HOST")
  S2_SESSION_HOST="$STAGE2_HOST/$SID"
  S2_SESSION_C="$STAGE2_C/$SID"
  T2_HOST="$TIMING_HOST/stage2/${SID}.csv"
  T2_C="$RUN_C/timing/stage2/${SID}.csv"
  mkdir -p "$S2_SESSION_HOST"

  printf '%s\t%s\n' "$GID" "$SID" >> "$SESSION_MAP"

  echo "------------------------------------------------------------"
  echo "Stage2 $idx/$total GID=$GID"
  echo "Stage1=$S1_SESSION_C"
  echo "Stage2=$S2_SESSION_C"

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
      --stage2_bundle "$STAGE2_BUNDLE_C" \
      --timing_csv "$T2_C" \
      --pole_candidate_threshold "$POLE_CANDIDATE_THRESHOLD" \
      --line_candidate_threshold "$LINE_CANDIDATE_THRESHOLD" \
      --line_weak_threshold "$LINE_WEAK_THRESHOLD" \
      --line_competition_ratio "$LINE_COMPETITION_RATIO" \
      --pole_min_voxels "$POLE_MIN_VOXELS" \
      --line_min_voxels "$LINE_MIN_VOXELS" \
      --edge_width_vox "$EDGE_WIDTH_VOX" \
      --resume 0 \
    2>&1 | tee "$LOG_HOST/stage2/${SID}.log"

  if [ ! -f "$S2_SESSION_HOST/inference_manifest.csv" ]; then
    echo "ERROR: Stage2 manifest missing for $GID" >&2
    exit 12
  fi
  printf 'OK %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$GID" > "$STATUS_HOST/${SID}.stage2.ok"
done

printf 'PHASE2_STAGE2_OK %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_HOST/PHASE2_STAGE2_OK.txt"

###############################################################################
# STAGE 3 - fresh full legal window for every newest slice
###############################################################################

echo "============================================================"
echo "STAGE 3 QUALITY RUN"
echo "============================================================"

idx=0
total=$(wc -l < "$SESSION_MAP" | tr -d ' ')
while IFS=$'\t' read -r GID SID; do
  [ -n "$GID" ] || continue
  idx=$((idx + 1))
  S2_SESSION_C="$STAGE2_C/$SID"
  S3_SESSION_HOST="$STAGE3_HOST/$SID"
  S3_SESSION_C="$STAGE3_C/$SID"
  mkdir -p "$S3_SESSION_HOST"

  echo "------------------------------------------------------------"
  echo "Stage3 $idx/$total GID=$GID"
  echo "Stage2=$S2_SESSION_C"
  echo "Stage3=$S3_SESSION_C"

  docker run --rm \
    --mount "type=bind,source=$EXP_REPO/v4,target=/workspace/v4,readonly" \
    --mount "type=bind,source=$EXP_REPO/ops/v4_stage23_quality,target=/workspace/quality,readonly" \
    --mount "type=bind,source=$HOST_OUTPUTS,target=$CONTAINER_OUTPUTS" \
    --workdir /workspace/v4 \
    -e PYTHONPATH=/workspace/v4 \
    -e PYTHONPYCACHEPREFIX=/tmp/pycache \
    -e MPLCONFIGDIR=/tmp/matplotlib \
    "$IMAGE" \
    python /workspace/quality/run_stage3_quality_session.py \
      --stage2_dir "$S2_SESSION_C" \
      --output_dir "$S3_SESSION_C" \
      --session_filter "$GID" \
      --stage3_script /workspace/v4/reconstruct_v4_stage3.py \
      --max_sequence_gap 9 \
      --slice_length_ft 50 \
      --max_span_length_ft 450 \
      --resume 0 \
      --disable_plots 1 \
    2>&1 | tee "$LOG_HOST/stage3/${SID}.log"

  if [ ! -f "$S3_SESSION_HOST/QUALITY_STAGE3_SESSION_COMPLETED.json" ]; then
    echo "ERROR: Stage3 completion marker missing for $GID" >&2
    exit 20
  fi
  printf 'OK %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$GID" > "$STATUS_HOST/${SID}.stage3.ok"
done < "$SESSION_MAP"

printf 'PHASE3_STAGE3_OK %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_HOST/PHASE3_STAGE3_OK.txt"

###############################################################################
# PACKAGE Stage2 and Stage3 separately
###############################################################################

PACKAGE_HOST="$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_stage23_quality/packages"
mkdir -p "$PACKAGE_HOST"
RUN_ID=$(basename "$RUN_HOST")
STAGE2_ARCHIVE="$PACKAGE_HOST/v4_stage2_recall_${RUN_ID}.tar.gz"
STAGE3_ARCHIVE="$PACKAGE_HOST/v4_stage3_quality_${RUN_ID}.tar.gz"

archive_one() {
  local src_name="$1"
  local marker="$2"
  local dst="$3"
  tar -czf "$dst" \
    --exclude='*.npz' \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='*.ckpt' \
    --exclude='*.onnx' \
    --exclude='*.engine' \
    --exclude='*.safetensors' \
    --exclude='*.joblib' \
    -C "$RUN_HOST" "$src_name" RUN_INFO.txt session_map.tsv "$marker"
  tar -tzf "$dst" > /dev/null
  sha256sum "$dst" > "${dst}.sha256"
}

archive_one stage2 PHASE2_STAGE2_OK.txt "$STAGE2_ARCHIVE"
archive_one stage3 PHASE3_STAGE3_OK.txt "$STAGE3_ARCHIVE"

printf '%s\n' "$STAGE2_ARCHIVE" > "$HOME/LATEST_V4_STAGE2_RECALL_ARCHIVE.txt"
printf '%s\n' "$STAGE3_ARCHIVE" > "$HOME/LATEST_V4_STAGE3_QUALITY_ARCHIVE.txt"
printf 'ALL_STAGE23_OK %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_HOST/ALL_STAGE23_OK.txt"

echo "============================================================"
echo "ALL STAGE2 + STAGE3 QUALITY WORK COMPLETED"
echo "Run root:       $RUN_HOST"
echo "Stage2 archive: $STAGE2_ARCHIVE"
echo "Stage3 archive: $STAGE3_ARCHIVE"
echo "============================================================"
