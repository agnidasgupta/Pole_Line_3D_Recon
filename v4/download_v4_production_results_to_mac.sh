#!/usr/bin/env bash
set -euo pipefail
REMOTE_HOST=${REMOTE_HOST:-nebius-va}
REMOTE_ROOT=${REMOTE_ROOT:-/workspace/voxel_poleline/outputs/poleline_voxel_run_session_groups/v4_realtime}
LOCAL_ROOT=${LOCAL_ROOT:-$HOME/Documents/VEG_Data/POLE_Voxel/v4_realtime_production_review}
mkdir -p "$LOCAL_ROOT"
# Keep final diagnostics, Stage-2 metrics, replay timings/manifests, and Stage-3 reconstruction CSVs/PNGs.
# Exclude model binaries, NPZ caches, and bulky Stage-2 mining per-slice caches.
rsync -avh --progress --partial --prune-empty-dirs \
  --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' --exclude='*.joblib' --exclude='*.pkl' --exclude='*.npz' \
  --exclude='stage2_mining/per_slice/' \
  --exclude='csv/' \
  "$REMOTE_HOST:$REMOTE_ROOT/" "$LOCAL_ROOT/"
echo "Downloaded to $LOCAL_ROOT"
