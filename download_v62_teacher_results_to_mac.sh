#!/usr/bin/env bash
set -euo pipefail
REMOTE_HOST="${REMOTE_HOST:-nebius-va}"
REMOTE_RUN="${REMOTE_RUN:-/workspace/voxel_poleline/outputs/poleline_voxel_run_session_groups/v62_teacher_recall/}"
LOCAL_RUN="${LOCAL_RUN:-$HOME/Documents/VEG_Data/POLE_Voxel/v62_teacher_recall_complete/}"
mkdir -p "$LOCAL_RUN"
rsync -avhL --progress --partial --prune-empty-dirs \
  --exclude='*.npz' \
  --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' \
  --exclude='*.joblib' --exclude='*.pkl' \
  --exclude='*.onnx' --exclude='*.engine' --exclude='*.safetensors' --exclude='*.tflite' \
  --exclude='__pycache__/' \
  "$REMOTE_HOST:$REMOTE_RUN" "$LOCAL_RUN"
echo "Downloaded non-model/non-NPZ results to: $LOCAL_RUN"
echo "Files: $(find "$LOCAL_RUN" -type f | wc -l | tr -d ' ')"
echo "Size:  $(du -sh "$LOCAL_RUN" | awk '{print $1}')"
