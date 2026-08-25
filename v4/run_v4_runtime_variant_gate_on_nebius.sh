#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HOST_OUTPUTS=${HOST_OUTPUTS:-/workspace/voxel_poleline/outputs}
HOST_DATA=${HOST_DATA:-/data}
IMAGE=${IMAGE:-va-v4-realtime:torch241-cu121}
OUT=${OUT:-/outputs/poleline_voxel_run_session_groups/v4_realtime/diagnostics}
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
  --mount type=bind,source="$HOST_DATA",target=/data,readonly \
  --workdir /workspace/voxel_poleline \
  "$IMAGE" bash -lc 'set -e
    mkdir -p '"$OUT"'
    python compare_v4_runtime_variants.py \
      --input_dir /data/voxel_csv_combined \
      --max_files '"${MAX_FILES:-32}"' \
      --model_path /outputs/poleline_voxel_run_session_groups/precision_v4/train/precision_best.pt \
      --calibration_json /outputs/poleline_voxel_run_session_groups/precision_v4/full_val/calibration.json \
      --batch_size '"${BATCH_SIZE:-12}"' \
      --amp '"${AMP:-bf16}"' \
      --score_tol '"${SCORE_TOL:-1e-4}"' \
      --output '"$OUT"'/runtime_variant_equivalence.json
    python select_v4_runtime_mode.py \
      --gate_json '"$OUT"'/runtime_variant_equivalence.json \
      --output_env '"$OUT"'/v4_runtime_mode.env
  '
HOST_ENV="$HOST_OUTPUTS/${OUT#/outputs/}/v4_runtime_mode.env"
echo "Selected runtime environment: $HOST_ENV"
cat "$HOST_ENV"
