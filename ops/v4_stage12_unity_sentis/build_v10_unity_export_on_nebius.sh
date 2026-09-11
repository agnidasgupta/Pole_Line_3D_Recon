#!/usr/bin/env bash
set -euo pipefail
REPO=${EXP_REPO:-/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10}
UNITY_DIR="$REPO/ops/v4_stage12_unity_sentis"
OPT_DIR="$REPO/ops/v4_stage2_stage1_electrical_v10_opt"
OUTPUTS=${HOST_OUTPUTS:-/workspace/voxel_poleline/outputs}
BASE_IMAGE=${BASE_IMAGE:-va-v4-realtime:torch241-cu121}
EXPORT_IMAGE=${EXPORT_IMAGE:-va-v10-onnx-export:onnx117-ort120}
MODEL=${MODEL_HOST:-$OUTPUTS/poleline_voxel_run_session_groups/precision_v4/train/precision_best.pt}
CAL=${CALIBRATION_HOST:-$OUTPUTS/poleline_voxel_run_session_groups/precision_v4/full_val/calibration.json}
BUNDLE=${STAGE2_BUNDLE_HOST:-$OUTPUTS/poleline_voxel_run_session_groups/v4_realtime/stage2_refiner/local_refiner_bundle.joblib}
STAGE1_BASE=${STAGE1_BASELINE_HOST:-$OUTPUTS/poleline_voxel_run_session_groups/v4_production/full_dataset_runs/d9977c39c443f5fa14f8/20260825T203403Z/stage1}
STAMP=${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT="$OUTPUTS/poleline_voxel_run_session_groups/v4_stage23_quality/unity_export_v10_$STAMP"
PACKAGE="$RUN_ROOT/unity_package/Assets/StreamingAssets/V10Stage12"
LOG="$RUN_ROOT/ONNX_EXPORT.log"
PROFILE="$RUN_ROOT/selection/selected_electrical_profile.json"

fail() { echo "ERROR: $*" >&2; exit 1; }
to_container() {
  case "$1" in
    "$OUTPUTS") printf '/outputs\n' ;;
    "$OUTPUTS"/*) printf '/outputs%s\n' "${1#$OUTPUTS}" ;;
    *) fail "path is outside HOST_OUTPUTS: $1" ;;
  esac
}
for path in "$REPO/.git" "$UNITY_DIR/export_v10_stage1_onnx.py" "$OPT_DIR/learn_velasco_stage1_electrical_profile_opt.py" "$MODEL" "$CAL" "$BUNDLE" "$STAGE1_BASE/VELASCO_CUT_CP__session1"; do
  [ -e "$path" ] || fail "missing $path"
done
mkdir -p "$RUN_ROOT/selection" "$PACKAGE" "$RUN_ROOT/unity_package/Assets/Scripts" "$RUN_ROOT/unity_package/Assets/Editor"
printf '%s\n' "$RUN_ROOT" >/home/agni/LATEST_V10_UNITY_EXPORT_RUN.txt
printf '%s\n' "$LOG" >/home/agni/LATEST_V10_UNITY_EXPORT_LOG.txt
exec > >(tee -a "$LOG") 2>&1
echo "V10 UNITY EXPORT run_root=$RUN_ROOT"

docker build \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  -f "$UNITY_DIR/Dockerfile.v10_onnx_export" \
  -t "$EXPORT_IMAGE" "$UNITY_DIR"

CAL_C=$(to_container "$CAL")
BUNDLE_C=$(to_container "$BUNDLE")
PROFILE_SESSION_C=$(to_container "$STAGE1_BASE/VELASCO_CUT_CP__session1")
RUN_C=$(to_container "$RUN_ROOT")
MODEL_C=$(to_container "$MODEL")

docker run --rm \
  --mount "type=bind,source=$REPO/v4,target=/workspace/v4,readonly" \
  --mount "type=bind,source=$OPT_DIR,target=/workspace/quality,readonly" \
  --mount "type=bind,source=$OUTPUTS,target=/outputs" \
  --workdir /workspace/v4 -e PYTHONPATH=/workspace/v4:/workspace/quality \
  "$BASE_IMAGE" python /workspace/quality/learn_velasco_stage1_electrical_profile_opt.py \
    --stage1_dir "$PROFILE_SESSION_C" --calibration_json "$CAL_C" \
    --session_filter VELASCO_CUT_CP/session1 --reference_min 0 --reference_max 19 \
    --fragment_min 20 --fragment_max 39 --output_dir "$RUN_C/selection"
[ -s "$PROFILE" ] || fail "profile was not created"

mapfile -t REAL_NPZ < <(find "$OUTPUTS/poleline_voxel_run_session_groups/dataset_hardneg_v4opt_uncompressed" -type f -name '*.npz' -size +0c | sort | head -4)
[ "${#REAL_NPZ[@]}" -gt 0 ] || fail "no real dataset NPZ files found for FP16 decision parity"
REAL_ARGS=()
for path in "${REAL_NPZ[@]}"; do REAL_ARGS+=(--real-npz "$(to_container "$path")"); done

docker run --rm --gpus all \
  --mount "type=bind,source=$REPO/v4,target=/workspace/v4,readonly" \
  --mount "type=bind,source=$UNITY_DIR,target=/workspace/unity,readonly" \
  --mount "type=bind,source=$OUTPUTS,target=/outputs" \
  --workdir /workspace/v4 -e PYTHONPATH=/workspace/v4 \
  "$EXPORT_IMAGE" python /workspace/unity/export_v10_stage1_onnx.py \
    --checkpoint "$MODEL_C" --calibration "$CAL_C" \
    --stage2-profile "$RUN_C/selection/selected_electrical_profile.json" \
    --stage2-bundle "$BUNDLE_C" --output-dir "$RUN_C/unity_package/Assets/StreamingAssets/V10Stage12" \
    --fp16 1 --verify 1 --trt-benchmark-runs "${TRT_BENCHMARK_RUNS:-20}" \
    --build-trt-plan "${BUILD_TRT_PLAN:-0}" "${REAL_ARGS[@]}"

cp "$UNITY_DIR/Assets/Scripts/V10Stage12SentisInferenceManager.cs" "$RUN_ROOT/unity_package/Assets/Scripts/"
cp "$UNITY_DIR/Assets/Editor/V10SentisFp16Quantizer.cs" "$RUN_ROOT/unity_package/Assets/Editor/"
cp "$UNITY_DIR/run_unity_stage1_csv_stage2.py" "$RUN_ROOT/unity_package/"
cp "$UNITY_DIR/README.md" "$RUN_ROOT/unity_package/"
find "$RUN_ROOT/unity_package" -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum >"$RUN_ROOT/unity_package/SHA256SUMS.txt"
printf 'completed_utc=%s\nrun_root=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RUN_ROOT" >"$RUN_ROOT/UNITY_EXPORT_COMPLETE.txt"
echo "V10_UNITY_EXPORT_COMPLETE run_root=$RUN_ROOT"
