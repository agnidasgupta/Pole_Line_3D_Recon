#!/usr/bin/env bash
set -euo pipefail
: "${SESSION_FILTER:?Set SESSION_FILTER=geography/sessionN}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HOST_OUTPUTS=${HOST_OUTPUTS:-/workspace/voxel_poleline/outputs}
IMAGE=${IMAGE:-va-v4-realtime:torch241-cu121}
RUNTIME_MODE_ENV=${RUNTIME_MODE_ENV:-$HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_realtime/diagnostics/v4_runtime_mode.env}
if [[ -z ${RUNTIME_MODE+x} && -f "$RUNTIME_MODE_ENV" ]]; then
  source "$RUNTIME_MODE_ENV"
  RUNTIME_MODE=${V4_RUNTIME_MODE:-full_cpu}
fi
RUNTIME_MODE=${RUNTIME_MODE:-full_cpu}
case "$RUNTIME_MODE" in
  full_cpu)   EVALUATE_ALL_CORES=1; GPU_COORD_CHANNELS=0 ;;
  active_cpu) EVALUATE_ALL_CORES=0; GPU_COORD_CHANNELS=0 ;;
  full_gpu)   EVALUATE_ALL_CORES=1; GPU_COORD_CHANNELS=1 ;;
  active_gpu) EVALUATE_ALL_CORES=0; GPU_COORD_CHANNELS=1 ;;
  *) echo "ERROR: RUNTIME_MODE must be full_cpu|active_cpu|full_gpu|active_gpu" >&2; exit 2 ;;
esac
SAFE=$(printf '%s' "$SESSION_FILTER" | tr '/ ' '__')
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-/outputs/poleline_voxel_run_session_groups/v4_realtime/benchmarks/${SAFE}_${RUNTIME_MODE}_${STAMP}}
NAME=${NAME:-poleline-v4-realtime-benchmark}
export SESSION_FILTER OUT NAME IMAGE EVALUATE_ALL_CORES GPU_COORD_CHANNELS
export RESUME=0 STAGE3_EVERY_SLICE=1 STAGE3_EXECUTION=inprocess FINALIZE_FULL_SESSION=0 WRITE_ROW_CSV=0
export AMP=${AMP:-bf16} BATCH_SIZE=${BATCH_SIZE:-12} COMPILE_MODEL=${COMPILE_MODEL:-0} MAX_SLICES=${MAX_SLICES:-0}
"$HERE/run_v4_realtime_session_on_nebius.sh"
set +e
code=$(sudo docker wait "$NAME")
set -e
if [[ "$code" != "0" ]]; then
  echo "ERROR: replay exited $code" >&2
  sudo docker logs --tail 300 "$NAME" >&2 || true
  exit "$code"
fi
sudo docker run --rm --gpus all \
  -e PYTHONPYCACHEPREFIX=/tmp/pycache -e MPLCONFIGDIR=/tmp/matplotlib -e XDG_CACHE_HOME=/tmp/xdg-cache -e TMPDIR=/tmp \
  --mount type=bind,source="$HERE",target=/workspace/voxel_poleline,readonly \
  --mount type=bind,source="$HOST_OUTPUTS",target=/outputs \
  --workdir /workspace/voxel_poleline \
  "$IMAGE" bash -lc "set -e
    python verify_v4_realtime_replay.py --replay_dir '$OUT' --max_span_slices 9 --max_span_length_ft 450
    python summarize_v4_realtime_timing.py --timing_csv '$OUT/realtime_slice_timing.csv' --output_json '$OUT/realtime_timing_summary.json' --output_txt '$OUT/realtime_timing_summary.txt'
  "
echo "V4_REALTIME_BENCHMARK_OK"
echo "Output: $OUT"
echo "Timing: $OUT/realtime_timing_summary.txt"
