#!/usr/bin/env bash
POLELINE_LEGACY_SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/upload_v62_to_nebius_from_mac.sh"
set -euo pipefail
HOST="${HOST:-nebius-va}"
REMOTE_PARENT="${REMOTE_PARENT:-/workspace/voxel_poleline}"
REPO_NAME="${REPO_NAME:-Pole_Line_3D_Recon}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${POLELINE_LEGACY_SOURCE}")" && pwd)"

rsync -avh --delete \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='*.npz' \
  --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' \
  --exclude='*.joblib' --exclude='*.pkl' \
  --exclude='*.onnx' --exclude='*.engine' \
  --exclude='*.safetensors' --exclude='*.tflite' \
  "$SCRIPT_DIR/" "$HOST:$REMOTE_PARENT/$REPO_NAME/"

echo "Synced code to $HOST:$REMOTE_PARENT/$REPO_NAME/"
