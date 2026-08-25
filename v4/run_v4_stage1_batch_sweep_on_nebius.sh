#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HOST_OUTPUTS=${HOST_OUTPUTS:-/workspace/voxel_poleline/outputs}; HOST_DATA=${HOST_DATA:-/data}; IMAGE=${IMAGE:-va-v4-realtime:torch241-cu121}
RUNTIME_MODE_ENV=${RUNTIME_MODE_ENV:-$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_realtime/diagnostics/v4_runtime_mode.env}
if [[ -f "$RUNTIME_MODE_ENV" ]]; then source "$RUNTIME_MODE_ENV"; fi
RUNTIME_MODE=${RUNTIME_MODE:-${V4_RUNTIME_MODE:-full_cpu}}
OUT=${OUT:-/outputs/poleline_voxel_run_session_groups/v4_realtime/diagnostics/batch_sweep_${RUNTIME_MODE}}
sudo docker run --rm --gpus all \
  -e PYTHONPYCACHEPREFIX=/tmp/pycache -e MPLCONFIGDIR=/tmp/matplotlib -e XDG_CACHE_HOME=/tmp/xdg-cache \
  -e TORCH_HOME=/tmp/torch-home -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor -e TRITON_CACHE_DIR=/tmp/triton \
  -e CUDA_CACHE_PATH=/tmp/cuda-cache -e NUMBA_CACHE_DIR=/tmp/numba-cache -e JOBLIB_TEMP_FOLDER=/tmp/joblib -e TMPDIR=/tmp \
  --mount type=bind,source="$HERE",target=/workspace/voxel_poleline,readonly \
  --mount type=bind,source="$HOST_OUTPUTS",target=/outputs \
  --mount type=bind,source="$HOST_DATA",target=/data,readonly \
  --workdir /workspace/voxel_poleline \
  "$IMAGE" python benchmark_v4_stage1_batch_sizes.py \
    --input_dir /data/voxel_csv_combined \
    --model_path /outputs/poleline_voxel_run_session_groups/precision_v4/train/precision_best.pt \
    --calibration_json /outputs/poleline_voxel_run_session_groups/precision_v4/full_val/calibration.json \
    --runtime_mode "$RUNTIME_MODE" --batch_sizes "${BATCH_SIZES:-8,12,16,20}" --max_files "${MAX_FILES:-8}" --amp "${AMP:-bf16}" --score_tol "${SCORE_TOL:-1e-4}" --output_dir "$OUT"
echo "Created $OUT/batch_size_summary.csv"
