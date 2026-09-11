#!/usr/bin/env bash
set -euo pipefail
[[ -f /.dockerenv ]] || { echo 'ERROR: this launcher must run inside the V4 Docker image' >&2; exit 2; }
DATASET_DIR=${DATASET_DIR:-/outputs/poleline_voxel_run_session_groups/dataset_hardneg_v4opt_uncompressed}
OUT=${OUT:-/outputs/poleline_voxel_run_session_groups/precision_v4_experimental}
python train_v4_stage1.py \
  --dataset_dir "$DATASET_DIR" \
  --output_dir "$OUT" \
  --resume_checkpoint "${RESUME_CHECKPOINT:-}" \
  --epochs "${EPOCHS:-30}" \
  --samples_per_epoch "${SAMPLES_PER_EPOCH:-12000}" \
  --eval_samples "${EVAL_SAMPLES:-3072}" \
  --batch_size "${BATCH_SIZE:-3}" \
  --grad_accum "${GRAD_ACCUM:-2}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --amp "${AMP:-bf16}" \
  --compile_model "${COMPILE_MODEL:-0}"
