#!/usr/bin/env bash
set -euo pipefail
WORK_DIR="${WORK_DIR:-/outputs/poleline_voxel_run_session_groups}"
RAW_INPUT_DIR="${RAW_INPUT_DIR:-/data/voxel_csv_combined}"
V4_DIR="${V4_DIR:-$WORK_DIR/precision_v4}"
V62_DIR="${V62_DIR:-$WORK_DIR/v62_teacher_recall}"
DATASET="$V62_DIR/dataset_local"
STAGE1="$V62_DIR/stage1_train"
CAND="$V62_DIR/candidate_full_val"
SELECTED="$V62_DIR/selected"
STAGE1_TEST="$V62_DIR/stage1_test"
MINE="$V62_DIR/stage2_component_mining"
REFINER="$V62_DIR/stage2_refiner"
TESTINF="$V62_DIR/test_inference"
DIAG="$V62_DIR/training_diagnostics"
mkdir -p "$V62_DIR" "$STAGE1" "$CAND" "$SELECTED" "$MINE" "$REFINER" "$DIAG"
LOG="$V62_DIR/v62_teacher_training_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
stage=preflight
fail(){ code=$?; python - "$V62_DIR" "$stage" "$code" "$LOG" <<'PY'
import json,os,sys,time
out,stage,code,log=sys.argv[1:]
json.dump({'completed':False,'stage':stage,'exit_code':int(code),'log':log,'failed_at':time.strftime('%F %T')},open(os.path.join(out,'TRAINING_FAILED.json'),'w'),indent=2)
PY
exit "$code"; }
trap fail ERR
V4_CKPT="${V4_TEACHER_CHECKPOINT:-$V4_DIR/train/precision_best.pt}"
V4_CAL="${V4_TEACHER_CALIBRATION:-$V4_DIR/full_val/calibration.json}"
test -d "$RAW_INPUT_DIR" || { echo "Raw input missing: $RAW_INPUT_DIR"; exit 2; }
test -f "$V4_CKPT" || { echo "V4 teacher checkpoint missing: $V4_CKPT"; exit 2; }

stage="prepare dataset"
python prepare_v62_dataset.py --input_dir "$RAW_INPUT_DIR" --output_dir "$DATASET" --val_frac "${VAL_FRAC:-0.15}" --test_frac "${TEST_FRAC:-0.15}" --resume 1
GRID=$(python - "$DATASET/manifests/summary.json" <<'PY'
import json,sys; print(*json.load(open(sys.argv[1]))['grid_size_xyz'])
PY
)

stage="Stage 1 V4-positive-teacher training"
train_args=(
  --dataset_dir "$DATASET" --resume_checkpoint "$V4_CKPT" --v4_teacher_checkpoint "$V4_CKPT"
  --output_dir "$STAGE1" --epochs "${EPOCHS:-30}" --samples_per_epoch "${SAMPLES_PER_EPOCH:-18000}"
  --eval_samples "${EVAL_SAMPLES:-3072}" --batch_size "${TRAIN_BATCH:-3}" --grad_accum "${GRAD_ACCUM:-2}"
  --num_workers "${NUM_WORKERS:-8}" --context_xy "${CONTEXT_XY:-256}" --context_z "${CONTEXT_Z:-128}"
  --compile_model "${TORCH_COMPILE:-1}" --pole_fp_prob 0 --line_fp_prob 0 --lambda_replay_fp 0
  --geometry_unlabeled_discount "${GEOMETRY_UNLABELED_DISCOUNT:-0.85}" --edge_asset_prob "${EDGE_ASSET_PROB:-0.15}"
  --edge_width_vox "${EDGE_WIDTH_VOX:-10}" --lambda_v4_line_teacher "${LAMBDA_V4_LINE_TEACHER:-0.30}"
  --v4_teacher_min_line_score "${V4_TEACHER_MIN_LINE_SCORE:-0.30}"
  --v4_teacher_line_over_pole_margin "${V4_TEACHER_LINE_OVER_POLE_MARGIN:-0.05}"
  --v4_teacher_vertical_margin "${V4_TEACHER_VERTICAL_MARGIN:-0.15}"
  --pole_target_precision "${STAGE1_POLE_TARGET_PRECISION:-0.80}" --pole_target_recall "${STAGE1_POLE_TARGET_RECALL:-0.85}"
  --pole_target_iou "${STAGE1_POLE_TARGET_IOU:-0.62}" --line_target_precision "${STAGE1_LINE_TARGET_PRECISION:-0.58}"
  --line_target_recall "${STAGE1_LINE_TARGET_RECALL:-0.88}" --line_target_iou "${STAGE1_LINE_TARGET_IOU:-0.52}"
  --line_recall_weight "${STAGE1_LINE_RECALL_WEIGHT:-2.5}"
)
[[ -f "$V4_CAL" ]] && train_args+=(--v4_teacher_calibration_json "$V4_CAL")
python train_v6_stage1.py "${train_args[@]}"

stage="Stage 1 exhaustive validation / selection"
python - "$STAGE1/candidate_manifest.json" > "$V62_DIR/candidate_list.txt" <<'PY'
import json,sys
for r in json.load(open(sys.argv[1]))['candidates']: print(f"{int(r['epoch']):03d} {r['path']}")
PY
while read -r epoch ckpt; do
  python full_scene_evaluate_v6_stage1.py --dataset_dir "$DATASET" --model_path "$ckpt" --output_dir "$CAND/epoch_$epoch" --split val \
    --patch_size 64 --context_xy "${CONTEXT_XY:-256}" --context_z "${CONTEXT_Z:-128}" --core_size 48 \
    --batch_size "${EVAL_BATCH:-5}" --build_workers "${BUILD_WORKERS:-6}" --resume 1 --compile_model "${TORCH_COMPILE:-1}" \
    --pole_target_precision "${STAGE1_POLE_TARGET_PRECISION:-0.80}" --pole_target_recall "${STAGE1_POLE_TARGET_RECALL:-0.85}" \
    --pole_target_iou "${STAGE1_POLE_TARGET_IOU:-0.62}" --line_target_precision "${STAGE1_LINE_TARGET_PRECISION:-0.58}" \
    --line_target_recall "${STAGE1_LINE_TARGET_RECALL:-0.88}" --line_target_iou "${STAGE1_LINE_TARGET_IOU:-0.52}" \
    --line_recall_weight "${STAGE1_LINE_RECALL_WEIGHT:-2.5}"
done < "$V62_DIR/candidate_list.txt"
python select_v6_candidate.py --train_dir "$STAGE1" --candidate_eval_root "$CAND" --output_dir "$SELECTED"
python full_scene_evaluate_v6_stage1.py --dataset_dir "$DATASET" --model_path "$SELECTED/v6_stage1_selected.pt" --output_dir "$STAGE1_TEST" --split test \
  --calibration_json "$SELECTED/calibration.json" --patch_size 64 --context_xy "${CONTEXT_XY:-256}" --context_z "${CONTEXT_Z:-128}" \
  --core_size 48 --batch_size "${EVAL_BATCH:-5}" --build_workers "${BUILD_WORKERS:-6}" --resume 1 --compile_model "${TORCH_COMPILE:-1}" \
  --pole_target_precision "${STAGE1_POLE_TARGET_PRECISION:-0.80}" --pole_target_recall "${STAGE1_POLE_TARGET_RECALL:-0.85}" \
  --pole_target_iou "${STAGE1_POLE_TARGET_IOU:-0.62}" --line_target_precision "${STAGE1_LINE_TARGET_PRECISION:-0.58}" \
  --line_target_recall "${STAGE1_LINE_TARGET_RECALL:-0.88}" --line_target_iou "${STAGE1_LINE_TARGET_IOU:-0.52}" \
  --line_recall_weight "${STAGE1_LINE_RECALL_WEIGHT:-2.5}"

stage="Stage 2 soft-gate component mining"
python mine_v62_local_components.py --dataset_dir "$DATASET" --model_path "$SELECTED/v6_stage1_selected.pt" --output_dir "$MINE" \
  --splits train,val --pole_candidate_threshold "${POLE_CANDIDATE_THRESHOLD:-0.15}" \
  --line_candidate_threshold "${LINE_CANDIDATE_THRESHOLD:-0.08}" --line_weak_threshold "${LINE_WEAK_THRESHOLD:-0.04}" \
  --line_competition_ratio "${LINE_COMPETITION_RATIO:-0.55}" --line_min_voxels "${LINE_MIN_VOXELS:-3}" \
  --edge_width_vox "${EDGE_WIDTH_VOX:-10}" --batch_size "${EVAL_BATCH:-5}" --build_workers "${BUILD_WORKERS:-6}" \
  --compile_model "${TORCH_COMPILE:-1}" --resume 1

stage="Stage 2 local refiner training"
python train_v62_local_refiners.py --components_csv "$MINE/all_local_components.csv.gz" --output_dir "$REFINER" \
  --pole_target_precision "${STAGE2_POLE_TARGET_PRECISION:-0.80}" --line_target_precision "${STAGE2_LINE_TARGET_PRECISION:-0.60}" \
  --pole_target_recall "${STAGE2_POLE_TARGET_RECALL:-0.95}" --line_target_recall "${STAGE2_LINE_TARGET_RECALL:-0.98}"

stage="held-out Stage 1+2 inference"
INPUT_DIR="$RAW_INPUT_DIR" OUTPUT_DIR="$TESTINF" MODEL_PATH="$SELECTED/v6_stage1_selected.pt" CALIBRATION_JSON="$SELECTED/calibration.json" \
LOCAL_REFINER_BUNDLE="$REFINER/local_refiner_bundle.joblib" MANIFEST_JSON="$DATASET/manifests/test.json" GRID_SIZE="$GRID" \
POLE_CANDIDATE_THRESHOLD="${POLE_CANDIDATE_THRESHOLD:-0.15}" LINE_CANDIDATE_THRESHOLD="${LINE_CANDIDATE_THRESHOLD:-0.08}" \
LINE_WEAK_THRESHOLD="${LINE_WEAK_THRESHOLD:-0.04}" LINE_COMPETITION_RATIO="${LINE_COMPETITION_RATIO:-0.55}" LINE_MIN_VOXELS="${LINE_MIN_VOXELS:-3}" \
./run_v62_inference.sh

stage="training diagnostics"
V4_TEST_METRICS="${V4_TEST_METRICS:-$V4_DIR/full_test/full_scene_metrics.json}"
if [[ -f "$V4_TEST_METRICS" ]]; then
  python compare_v4_v62_line_metrics.py --v4_metrics "$V4_TEST_METRICS" --v62_metrics "$STAGE1_TEST/full_scene_metrics.json" --output_dir "$DIAG"
else
  echo "V4 test metrics not found at $V4_TEST_METRICS; skipping comparison plot"
fi
python - "$V62_DIR" "$LOG" <<'PY'
import json,os,sys,time
out,log=sys.argv[1:]
json.dump({'completed':True,'completed_at':time.strftime('%F %T'),'log':log,'teacher_distillation':True,'soft_stage2_line_gating':True},open(os.path.join(out,'TRAINING_COMPLETED.json'),'w'),indent=2)
PY
rm -f "$V62_DIR/TRAINING_FAILED.json"
echo "V6.2 TEACHER-RECALL TRAINING COMPLETE"
