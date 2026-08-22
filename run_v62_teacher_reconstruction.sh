#!/usr/bin/env bash
set -euo pipefail
WORK_DIR="${WORK_DIR:-/outputs/poleline_voxel_run_session_groups}"
RAW_INPUT_DIR="${RAW_INPUT_DIR:-/data/voxel_csv_combined}"
V62_DIR="${V62_DIR:-$WORK_DIR/v62_teacher_recall}"
INFERENCE_DIR="${INFERENCE_DIR:-$V62_DIR/inference_all}"
STAGE3="${OUTPUT_DIR:-$V62_DIR/stage3_reconstruction}"
test -d "$INFERENCE_DIR/stage2_objects" || { echo "Missing Stage-2 inference objects: $INFERENCE_DIR/stage2_objects"; exit 2; }
test -s "$INFERENCE_DIR/inference_manifest.csv" || { echo "Missing/empty inference manifest: $INFERENCE_DIR/inference_manifest.csv"; exit 2; }
mkdir -p "$(dirname "$STAGE3")"
RESUME_STAGE3="${RESUME_STAGE3:-0}"
if [[ "$RESUME_STAGE3" == "1" ]]; then
  mkdir -p "$STAGE3"
  rm -f "$STAGE3/COMPLETED.json"
  echo "[stage3-wrapper] resume enabled: preserving completed session directories in $STAGE3"
elif [[ -d "$STAGE3" && "${BACKUP_EXISTING_STAGE3:-1}" == "1" ]] && find "$STAGE3" -mindepth 1 -print -quit | grep -q .; then
  stamp=$(date +%Y%m%d_%H%M%S); mv "$STAGE3" "${STAGE3}_before_rerun_${stamp}"
fi
INFERENCE_DIR="$INFERENCE_DIR" METADATA_DIR="$RAW_INPUT_DIR" OUTPUT_DIR="$STAGE3" \
WORLD_UNITS_TO_FT="${WORLD_UNITS_TO_FT:-0.5}" MIN_POLE_SEPARATION_FT="${MIN_POLE_SEPARATION_FT:-10}" \
MAX_SPAN_LENGTH_FT="${MAX_SPAN_LENGTH_FT:-450}" MAX_SPAN_SLICES="${MAX_SPAN_SLICES:-9}" \
ALLOWED_POLE_HEIGHT_VARIATION_FT="${ALLOWED_POLE_HEIGHT_VARIATION_FT:-8}" MAX_POLE_HEIGHT_ADJUST_FT="${MAX_POLE_HEIGHT_ADJUST_FT:-8}" \
SPAN_COMPLETION_MAX_ANGLE_DEG="${SPAN_COMPLETION_MAX_ANGLE_DEG:-12}" SPAN_COMPLETION_MAX_CONNECTOR_ANGLE_DEG="${SPAN_COMPLETION_MAX_CONNECTOR_ANGLE_DEG:-15}" \
SPAN_COMPLETION_MAX_LATERAL_FT="${SPAN_COMPLETION_MAX_LATERAL_FT:-1.5}" SPAN_COMPLETION_MAX_Z_ERROR_FT="${SPAN_COMPLETION_MAX_Z_ERROR_FT:-3}" \
SPAN_COMPLETION_MAX_GAP_FT="${SPAN_COMPLETION_MAX_GAP_FT:-100}" HIDDEN_POLE_MIN_LINE_ANGLE_DEG="${HIDDEN_POLE_MIN_LINE_ANGLE_DEG:-20}" \
HIDDEN_POLE_MAX_ENDPOINT_EXTRAP_FT="${HIDDEN_POLE_MAX_ENDPOINT_EXTRAP_FT:-20}" HIDDEN_POLE_MIN_TRACK_EVIDENCE="${HIDDEN_POLE_MIN_TRACK_EVIDENCE:-2.0}" \
HIDDEN_POLE_REQUIRE_DISTINCT_ANCHOR_POLES="${HIDDEN_POLE_REQUIRE_DISTINCT_ANCHOR_POLES:-1}" \
RESUME_STAGE3="$RESUME_STAGE3" \
./run_v62_stage3.sh

# Refresh combined diagnostics when the prerequisite evaluation artifacts exist.
if [[ -f "$V62_DIR/stage1_test/full_scene_metrics.json" && -f "$V62_DIR/stage2_refiner/local_refiner_metrics.json" && -f "$V62_DIR/inference_all/inference_metrics.json" && -f "$STAGE3/COMPLETED.json" ]]; then
  mkdir -p "$V62_DIR/diagnostics"
  python collect_v62_diagnostics.py \
    --stage1_test "$V62_DIR/stage1_test/full_scene_metrics.json" \
    --stage2_metrics "$V62_DIR/stage2_refiner/local_refiner_metrics.json" \
    --inference_metrics "$V62_DIR/inference_all/inference_metrics.json" \
    --stage3_completed "$STAGE3/COMPLETED.json" \
    --output_dir "$V62_DIR/diagnostics"
fi

echo "V6.2 TEACHER-RECALL RECONSTRUCTION COMPLETE"
