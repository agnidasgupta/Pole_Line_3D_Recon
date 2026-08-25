#!/usr/bin/env bash
set -euo pipefail
REMOTE_HOST=${REMOTE_HOST:-nebius-va}
REMOTE_ROOT=${REMOTE_ROOT:-/workspace/voxel_poleline/outputs/poleline_voxel_run_session_groups/v4_realtime}
LOCAL_ROOT=${LOCAL_ROOT:-$HOME/Documents/VEG_Data/POLE_Voxel/v4_realtime}
mkdir -p "$LOCAL_ROOT"
# Metrics/diagnostics, excluding trained model binaries by default.
rsync -avh --progress --partial --prune-empty-dirs \
  --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' --exclude='*.joblib' --exclude='*.pkl' --exclude='*.npz' \
  --exclude='stage2_mining/per_slice/' \
  "$REMOTE_HOST:$REMOTE_ROOT/" "$LOCAL_ROOT/"
echo "Downloaded to $LOCAL_ROOT"
