#!/usr/bin/env bash
set -euo pipefail
REPO=${EXP_REPO:-/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10}
OUTPUTS=${HOST_OUTPUTS:-/workspace/voxel_poleline/outputs}
IMAGE=${IMAGE:-va-v4-realtime:torch241-cu121}
STAMP=${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT_HOST=${OUT_HOST:-$OUTPUTS/poleline_voxel_run_session_groups/precision_v4_experimental_$STAMP}
LOG=${LOG:-/home/agni/V4_STAGE1_TRAINING_$STAMP.log}
mkdir -p "$OUT_HOST"
nohup docker run --rm \
  --gpus all \
  --mount "type=bind,source=$REPO/v4,target=/workspace/v4,readonly" \
  --mount "type=bind,source=$OUTPUTS,target=/outputs" \
  --workdir /workspace/v4 \
  -e DATASET_DIR="${DATASET_DIR:-/outputs/poleline_voxel_run_session_groups/dataset_hardneg_v4opt_uncompressed}" \
  -e OUT="/outputs${OUT_HOST#$OUTPUTS}" \
  -e RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}" \
  -e EPOCHS="${EPOCHS:-30}" \
  -e SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-12000}" \
  -e EVAL_SAMPLES="${EVAL_SAMPLES:-3072}" \
  -e BATCH_SIZE="${BATCH_SIZE:-3}" \
  -e GRAD_ACCUM="${GRAD_ACCUM:-2}" \
  -e NUM_WORKERS="${NUM_WORKERS:-8}" \
  -e AMP="${AMP:-bf16}" \
  -e COMPILE_MODEL="${COMPILE_MODEL:-0}" \
  "$IMAGE" bash /workspace/v4/run_v4_stage1_training.sh \
  >"$LOG" 2>&1 </dev/null &
PID=$!
printf '%s\n' "$PID" >/home/agni/LATEST_V4_STAGE1_TRAINING_PID.txt
printf '%s\n' "$LOG" >/home/agni/LATEST_V4_STAGE1_TRAINING_LOG.txt
printf '%s\n' "$OUT_HOST" >/home/agni/LATEST_V4_STAGE1_TRAINING_RUN.txt
printf 'STARTED pid=%s output=%s log=%s\n' "$PID" "$OUT_HOST" "$LOG"
