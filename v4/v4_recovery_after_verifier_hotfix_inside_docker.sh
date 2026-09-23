#!/usr/bin/env bash
set -euo pipefail
[[ -f /.dockerenv ]] || { echo 'ERROR: recovery body must run inside Docker' >&2; exit 2; }
if [[ $# -ne 8 ]]; then
  echo "usage: $0 REC_ROOT OLD_TEST_ROOT SESSION MODEL CAL STAGE2 DIAG DEPLOY_FINGERPRINT" >&2
  exit 2
fi
REC=$1; OLD=$2; SESSION=$3; MODEL=$4; CAL=$5; STAGE2=$6; DIAG=$7; DEPLOY_FINGERPRINT=$8
mkdir -p "$REC" "$DIAG"
FAIL="$REC/FAILED.txt"
trap 'rc=$?; printf "FAILED_UTC=%s\nEXIT_CODE=%s\nLAST_COMMAND=%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$BASH_COMMAND" > "$FAIL"; exit "$rc"' ERR
rm -f "$FAIL"

echo '[recovery] validate patched code and verifier regression'
bash v4_code_validation_inside_docker.sh

echo '[recovery] re-verify already completed quick replay with corrected verifier'
python verify_v4_realtime_replay.py --replay_dir "$OLD/replay_quick" --max_sequence_gap 9 --slice_length_ft 50 --max_span_length_ft 450
python summarize_v4_realtime_timing.py --timing_csv "$OLD/replay_quick/realtime_slice_timing.csv" --output_json "$REC/quick_reused_timing_summary.json" --output_txt "$REC/quick_reused_timing_summary.txt"
cp -f "$OLD/replay_quick/REALTIME_REPLAY_VERIFICATION.json" "$REC/quick_reused_REALTIME_REPLAY_VERIFICATION.json"

source "$REC/v4_runtime_mode.env"
source "$REC/v4_batch_size.env"
case "$V4_RUNTIME_MODE" in
  full_cpu) EAC=1; GCC=0; FBS=0;;
  active_cpu) EAC=0; GCC=0; FBS=1;;
  full_gpu) EAC=1; GCC=1; FBS=0;;
  active_gpu) EAC=0; GCC=1; FBS=1;;
  *) echo "invalid selected runtime: $V4_RUNTIME_MODE" >&2; exit 9;;
esac

echo "[recovery] run full selected-session replay runtime=$V4_RUNTIME_MODE batch=$V4_BATCH_SIZE"
FULL="$REC/replay_full"
python run_v4_realtime_session.py \
  --input_dir /data/voxel_csv_combined --session_filter "$SESSION" --output_dir "$FULL" \
  --model_path "$MODEL" --calibration_json "$CAL" --stage2_bundle "$STAGE2" \
  --batch_size "$V4_BATCH_SIZE" --amp "${AMP:-bf16}" \
  --evaluate_all_cores "$EAC" --gpu_coord_channels "$GCC" --fixed_batch_shape "$FBS" \
  --max_sequence_gap 9 --slice_length_ft 50 --max_span_length_ft 450 \
  --stage3_every_slice 1 --stage3_execution inprocess --stage3_inmemory_cache 1 \
  --resume 1 --max_slices 0
python verify_v4_realtime_replay.py --replay_dir "$FULL" --max_sequence_gap 9 --slice_length_ft 50 --max_span_length_ft 450
python summarize_v4_realtime_timing.py --timing_csv "$FULL/realtime_slice_timing.csv" --output_json "$FULL/realtime_timing_summary.json" --output_txt "$FULL/realtime_timing_summary.txt"

# The host wrapper proved the old gate applies to byte-identical runtime/gate code
# and identical model/calibration/refiner artifacts. Promote that evidence to the
# current verifier-tooling fingerprint only after the new full replay passes.
cp -f "$REC/v4_runtime_mode.env" "$DIAG/v4_runtime_mode.env"
cp -f "$REC/v4_batch_size.env" "$DIAG/v4_batch_size.env"
cp -f "$REC/runtime_variant_equivalence.json" "$DIAG/runtime_variant_equivalence.json"
if [[ -d "$REC/batch_sweep" ]]; then rm -rf "$DIAG/batch_sweep"; cp -a "$REC/batch_sweep" "$DIAG/batch_sweep"; fi
printf '%s\n' "$DEPLOY_FINGERPRINT" > "$DIAG/gated_deployment_sha256.txt"
printf '%s\n' "$DEPLOY_FINGERPRINT" > "$DIAG/production_acceptance_deployment_sha256.txt"
cat > "$DIAG/PRODUCTION_ACCEPTANCE_OK.txt" <<EOF
V4_PRODUCTION_ACCEPTANCE_OK
accepted_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
session=$SESSION
recovery_root=$REC
runtime_mode=$V4_RUNTIME_MODE
batch_size=$V4_BATCH_SIZE
deployment_fingerprint=$DEPLOY_FINGERPRINT
max_sequence_gap=9
slice_length_ft=50
max_span_length_ft=450
max_observed_slice_centers=10
quick_replay_reverified=passed
full_session_tested=1
independent_stage_replay=reused_from_prior_pass
runtime_model_identity_guard=passed
replay_verifier_hotfix=passed
EOF
cat > "$REC/RECOVERY_ACCEPTANCE_OK.txt" <<EOF
V4_RECOVERY_ACCEPTANCE_OK
accepted_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
full_replay=$FULL
runtime_mode=$V4_RUNTIME_MODE
batch_size=$V4_BATCH_SIZE
deployment_fingerprint=$DEPLOY_FINGERPRINT
EOF
rm -f "$FAIL"
echo "V4_PRODUCTION_ACCEPTANCE_OK runtime=$V4_RUNTIME_MODE batch=$V4_BATCH_SIZE deployment=$DEPLOY_FINGERPRINT recovery_root=$REC"
