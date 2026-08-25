#!/usr/bin/env bash
set -euo pipefail
[[ -f /.dockerenv ]] || { echo 'ERROR: run_v4_stage2_training.sh is container-internal. On Nebius use run_v4_stage2_training_on_nebius.sh.' >&2; exit 2; }
DATASET_DIR=${DATASET_DIR:-/outputs/poleline_voxel_run_session_groups/dataset_hardneg_v4opt_uncompressed}
OUT=${OUT:-/outputs/poleline_voxel_run_session_groups/v4_realtime}
MODEL=${MODEL:-/outputs/poleline_voxel_run_session_groups/precision_v4/train/precision_best.pt}
CAL=${CAL:-/outputs/poleline_voxel_run_session_groups/precision_v4/full_val/calibration.json}
mkdir -p "$OUT/stage2_mining" "$OUT/stage2_refiner"
python mine_v4_stage2_components.py \
  --dataset_dir "$DATASET_DIR" --output_dir "$OUT/stage2_mining" \
  --model_path "$MODEL" --calibration_json "$CAL" \
  --batch_size "${BATCH_SIZE:-12}" --amp "${AMP:-bf16}" --compile_model "${COMPILE_MODEL:-0}" --evaluate_all_cores "${EVALUATE_ALL_CORES:-1}" --gpu_coord_channels "${GPU_COORD_CHANNELS:-0}" --resume "${RESUME:-1}" \
  --pole_candidate_threshold "${POLE_CANDIDATE_THRESHOLD:-0.15}" \
  --line_candidate_threshold "${LINE_CANDIDATE_THRESHOLD:-0.08}" \
  --line_weak_threshold "${LINE_WEAK_THRESHOLD:-0.04}" \
  --line_competition_ratio "${LINE_COMPETITION_RATIO:-0.55}"
python train_v4_stage2_refiners.py \
  --components_csv "$OUT/stage2_mining/components.csv" --output_dir "$OUT/stage2_refiner" \
  --pole_target_recall "${POLE_TARGET_RECALL:-0.95}" --line_target_recall "${LINE_TARGET_RECALL:-0.98}" \
  --pole_target_precision "${POLE_TARGET_PRECISION:-0.80}" --line_target_precision "${LINE_TARGET_PRECISION:-0.60}"
python - <<PY
import json, pathlib
p=pathlib.Path('$OUT/STAGE2_TRAINING_COMPLETED.json'); p.write_text(json.dumps({'completed':True,'model':'$MODEL','calibration':'$CAL','dataset':'$DATASET_DIR'},indent=2))
PY
