#!/usr/bin/env bash
set -euo pipefail
: "${REPLAY:?Set REPLAY to container-style path under /outputs, e.g. /outputs/poleline_voxel_run_session_groups/v4_realtime/replays/Geo_session1}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HOST_OUTPUTS=${HOST_OUTPUTS:-/workspace/voxel_poleline/outputs}; IMAGE=${IMAGE:-va-v4-realtime:torch241-cu121}
sudo docker run --rm --gpus all \
  -e PYTHONPYCACHEPREFIX=/tmp/pycache -e MPLCONFIGDIR=/tmp/matplotlib -e XDG_CACHE_HOME=/tmp/xdg-cache -e TMPDIR=/tmp \
  --mount type=bind,source="$HERE",target=/workspace/voxel_poleline,readonly \
  --mount type=bind,source="$HOST_OUTPUTS",target=/outputs \
  --workdir /workspace/voxel_poleline \
  "$IMAGE" bash -lc "set -e
    python verify_v4_realtime_replay.py --replay_dir '$REPLAY' --max_span_slices 9 --max_span_length_ft 450
    python summarize_v4_realtime_timing.py --timing_csv '$REPLAY/realtime_slice_timing.csv' --output_json '$REPLAY/realtime_timing_summary.json' --output_txt '$REPLAY/realtime_timing_summary.txt'
    cat '$REPLAY/realtime_timing_summary.txt'
  "
