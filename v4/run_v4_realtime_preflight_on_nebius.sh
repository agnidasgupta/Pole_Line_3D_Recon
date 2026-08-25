#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
HOST_OUTPUTS=${HOST_OUTPUTS:-/workspace/voxel_poleline/outputs}
IMAGE=${IMAGE:-va-v4-realtime:torch241-cu121}
MODEL=${MODEL:-/outputs/poleline_voxel_run_session_groups/precision_v4/train/precision_best.pt}
CAL=${CAL:-/outputs/poleline_voxel_run_session_groups/precision_v4/full_val/calibration.json}
DATASET_DIR=${DATASET_DIR:-/outputs/poleline_voxel_run_session_groups/dataset_hardneg_v4opt_uncompressed}
# V4_REALTIME_WRITABLE_CACHE_ENV_V2
sudo docker run --rm --gpus all \
  -e PYTHONPYCACHEPREFIX=/tmp/pycache \
  -e MPLCONFIGDIR=/tmp/matplotlib \
  -e XDG_CACHE_HOME=/tmp/xdg-cache \
  -e TORCH_HOME=/tmp/torch-home \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor \
  -e TRITON_CACHE_DIR=/tmp/triton \
  -e CUDA_CACHE_PATH=/tmp/cuda-cache \
  -e NUMBA_CACHE_DIR=/tmp/numba-cache \
  -e JOBLIB_TEMP_FOLDER=/tmp/joblib \
  -e TMPDIR=/tmp \
  --mount type=bind,source="$HERE",target=/workspace/voxel_poleline,readonly \
  --mount type=bind,source="$HOST_OUTPUTS",target=/outputs \
  --workdir /workspace/voxel_poleline \
  "$IMAGE" bash -lc "set -e; test -s '$MODEL'; test -s '$CAL'; test -s '$DATASET_DIR/manifests/train.json'; test -s '$DATASET_DIR/manifests/val.json'; python -m compileall -q .; python validate_v4_production_source_contract.py; python smoke_test_v4_realtime.py; python smoke_test_v4_stage2_runtime.py; python smoke_test_v4_stage3_incremental.py; python - <<'PY'
import torch
print('GPU=',torch.cuda.get_device_name(0)); print('torch=',torch.__version__,'cuda=',torch.version.cuda)
PY
 echo V4_REALTIME_NEBIUS_PREFLIGHT_OK"
