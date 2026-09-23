#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 9 ]]; then
  echo "usage: $0 TEST_ROOT SESSION MODEL CAL STAGE2 DIAG MAX_QUICK FULL_SESSION DEPLOY_FINGERPRINT" >&2
  exit 2
fi
TEST_ROOT=$1; SESSION=$2; MODEL=$3; CAL=$4; STAGE2=$5; DIAG=$6; MAX_QUICK=$7; FULL_SESSION=$8; DEPLOY_FINGERPRINT=$9
mkdir -p "$TEST_ROOT" "$DIAG"
FAIL="$TEST_ROOT/FAILED.txt"
trap 'rc=$?; printf "FAILED_UTC=%s\nEXIT_CODE=%s\nLAST_COMMAND=%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$BASH_COMMAND" > "$FAIL"; exit "$rc"' ERR
rm -f "$FAIL"

echo "[acceptance] source/static + synthetic smoke tests"
bash v4_code_validation_inside_docker.sh

echo "[acceptance] H100 Stage1+Stage2 runtime equivalence gate"
python compare_v4_runtime_variants.py --input_dir /data/voxel_csv_combined --max_files "${GATE_FILES:-32}" --model_path "$MODEL" --calibration_json "$CAL" --stage2_bundle "$STAGE2" --batch_size 12 --amp "${AMP:-bf16}" --score_tol "${SCORE_TOL:-1e-4}" --output "$TEST_ROOT/runtime_variant_equivalence.json"
python validate_v4_runtime_gate.py --gate_json "$TEST_ROOT/runtime_variant_equivalence.json"
python select_v4_runtime_mode.py --gate_json "$TEST_ROOT/runtime_variant_equivalence.json" --output_env "$TEST_ROOT/v4_runtime_mode.env"
source "$TEST_ROOT/v4_runtime_mode.env"

echo "[acceptance] selected-runtime batch sweep with Stage2 equivalence"
python benchmark_v4_stage1_batch_sizes.py --input_dir /data/voxel_csv_combined --model_path "$MODEL" --calibration_json "$CAL" --stage2_bundle "$STAGE2" --runtime_mode "$V4_RUNTIME_MODE" --batch_sizes "${BATCH_SIZES:-8,12,16,20,24,32}" --reference_batch 12 --max_files "${BATCH_SWEEP_FILES:-8}" --amp "${AMP:-bf16}" --score_tol "${SCORE_TOL:-1e-4}" --output_dir "$TEST_ROOT/batch_sweep"
python select_v4_batch_size.py --summary_json "$TEST_ROOT/batch_sweep/batch_size_summary.json" --output_env "$TEST_ROOT/v4_batch_size.env"
source "$TEST_ROOT/v4_batch_size.env"
case "$V4_RUNTIME_MODE" in
 full_cpu) EAC=1; GCC=0; FBS=0;;
 active_cpu) EAC=0; GCC=0; FBS=1;;
 full_gpu) EAC=1; GCC=1; FBS=0;;
 active_gpu) EAC=0; GCC=1; FBS=1;;
 *) echo "invalid selected runtime: $V4_RUNTIME_MODE" >&2; exit 9;;
esac

echo "[acceptance] independent durable Stage1 -> relocated Stage1 -> Stage2 -> relocated Stage2 -> Stage3"
S1="$TEST_ROOT/independent_stage1"
S1_RELOC="$TEST_ROOT/independent_stage1_relocated"
S2="$TEST_ROOT/independent_stage2"
S2_RELOC="$TEST_ROOT/independent_stage2_relocated"
S3="$TEST_ROOT/independent_stage3"
python run_v4_stage1.py --input_dir /data/voxel_csv_combined --session_filter "$SESSION" --output_dir "$S1" --model_path "$MODEL" --calibration_json "$CAL" --batch_size "$V4_BATCH_SIZE" --amp "${AMP:-bf16}" --evaluate_all_cores "$EAC" --gpu_coord_channels "$GCC" --fixed_batch_shape "$FBS" --resume 0 --max_slices 3
test -s "$S1/STAGE1_COMPLETED.json"
cp -a "$S1" "$S1_RELOC"
rm -rf "$S1"
python run_v4_stage2.py --stage1_dir "$S1_RELOC" --output_dir "$S2" --session_filter "$SESSION" --stage2_bundle "$STAGE2" --resume 0 --max_slices 3
test -s "$S2/STAGE2_COMPLETED.json"
cp -a "$S2" "$S2_RELOC"
rm -rf "$S2"
python run_v4_stage3.py --stage2_dir "$S2_RELOC" --output_dir "$S3" --session_filter "$SESSION" --max_sequence_gap 9 --slice_length_ft 50 --max_span_length_ft 450 --latest_only 0 --resume 0
test -s "$S3/STAGE3_COMPLETED.json"
# A second Stage3 output root proves reconstruction can be regenerated solely from the durable Stage2 boundary.
python run_v4_stage3.py --stage2_dir "$S2_RELOC" --output_dir "$TEST_ROOT/independent_stage3_repeat" --session_filter "$SESSION" --max_sequence_gap 9 --slice_length_ft 50 --max_span_length_ft 450 --latest_only 0 --resume 0
test -s "$TEST_ROOT/independent_stage3_repeat/STAGE3_COMPLETED.json"

echo "[acceptance] quick rolling realtime replay"
QUICK="$TEST_ROOT/replay_quick"
python run_v4_realtime_session.py --input_dir /data/voxel_csv_combined --session_filter "$SESSION" --output_dir "$QUICK" --model_path "$MODEL" --calibration_json "$CAL" --stage2_bundle "$STAGE2" --batch_size "$V4_BATCH_SIZE" --amp "${AMP:-bf16}" --evaluate_all_cores "$EAC" --gpu_coord_channels "$GCC" --fixed_batch_shape "$FBS" --max_sequence_gap 9 --slice_length_ft 50 --max_span_length_ft 450 --stage3_every_slice 1 --stage3_execution inprocess --stage3_inmemory_cache 1 --resume 0 --max_slices "$MAX_QUICK"
python verify_v4_realtime_replay.py --replay_dir "$QUICK" --max_sequence_gap 9 --slice_length_ft 50 --max_span_length_ft 450
python summarize_v4_realtime_timing.py --timing_csv "$QUICK/realtime_slice_timing.csv" --output_json "$QUICK/realtime_timing_summary.json" --output_txt "$QUICK/realtime_timing_summary.txt"

if [[ "$FULL_SESSION" == 1 ]]; then
 echo "[acceptance] full selected-session rolling replay"
 FULL="$TEST_ROOT/replay_full"
 python run_v4_realtime_session.py --input_dir /data/voxel_csv_combined --session_filter "$SESSION" --output_dir "$FULL" --model_path "$MODEL" --calibration_json "$CAL" --stage2_bundle "$STAGE2" --batch_size "$V4_BATCH_SIZE" --amp "${AMP:-bf16}" --evaluate_all_cores "$EAC" --gpu_coord_channels "$GCC" --fixed_batch_shape "$FBS" --max_sequence_gap 9 --slice_length_ft 50 --max_span_length_ft 450 --stage3_every_slice 1 --stage3_execution inprocess --stage3_inmemory_cache 1 --resume 0 --max_slices 0
 python verify_v4_realtime_replay.py --replay_dir "$FULL" --max_sequence_gap 9 --slice_length_ft 50 --max_span_length_ft 450
 python summarize_v4_realtime_timing.py --timing_csv "$FULL/realtime_slice_timing.csv" --output_json "$FULL/realtime_timing_summary.json" --output_txt "$FULL/realtime_timing_summary.txt"
fi

# Promotion files are copied only after every preceding command has passed.
cp "$TEST_ROOT/v4_runtime_mode.env" "$DIAG/v4_runtime_mode.env"
cp "$TEST_ROOT/v4_batch_size.env" "$DIAG/v4_batch_size.env"
cp "$TEST_ROOT/runtime_variant_equivalence.json" "$DIAG/runtime_variant_equivalence.json"
rm -rf "$DIAG/batch_sweep"; cp -a "$TEST_ROOT/batch_sweep" "$DIAG/batch_sweep"
printf '%s\n' "$DEPLOY_FINGERPRINT" > "$DIAG/gated_deployment_sha256.txt"
printf '%s\n' "$DEPLOY_FINGERPRINT" > "$DIAG/production_acceptance_deployment_sha256.txt"
cat > "$DIAG/PRODUCTION_ACCEPTANCE_OK.txt" <<MARK
V4_PRODUCTION_ACCEPTANCE_OK
accepted_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
session=$SESSION
test_root=$TEST_ROOT
runtime_mode=$V4_RUNTIME_MODE
batch_size=$V4_BATCH_SIZE
code_fingerprint=$(bash v4_code_fingerprint.sh)
deployment_fingerprint=$DEPLOY_FINGERPRINT
full_session_tested=$FULL_SESSION
max_sequence_gap=9
slice_length_ft=50
max_span_length_ft=450
max_observed_slice_centers=10
independent_stage_replay=passed
MARK
rm -f "$FAIL"
echo "V4_PRODUCTION_ACCEPTANCE_OK runtime=$V4_RUNTIME_MODE batch=$V4_BATCH_SIZE deployment=$DEPLOY_FINGERPRINT test_root=$TEST_ROOT"
