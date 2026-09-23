#!/usr/bin/env bash
# Verify the newest automatically discovered replay. No replay path is requested.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/v4_nebius_common.sh"
TEST_ROOT_HOST=$(v4_latest_test_root_host 2>/dev/null || true)
REPLAY_HOST=''
if [[ -n "$TEST_ROOT_HOST" && -s "$TEST_ROOT_HOST/replay_full/realtime_slice_timing.csv" ]]; then REPLAY_HOST="$TEST_ROOT_HOST/replay_full"
elif [[ -n "$TEST_ROOT_HOST" && -s "$TEST_ROOT_HOST/replay_quick/realtime_slice_timing.csv" ]]; then REPLAY_HOST="$TEST_ROOT_HOST/replay_quick"
elif [[ -s "$V4_RUN_ROOT_HOST/realtime_slice_timing.csv" ]]; then REPLAY_HOST="$V4_RUN_ROOT_HOST"
fi
[[ -n "$REPLAY_HOST" ]] || { echo "ERROR: no replay auto-discovered" >&2; exit 2; }
REPLAY=$(v4_to_container_output "$REPLAY_HOST")
v4_docker_run "python verify_v4_realtime_replay.py --replay_dir '$REPLAY' --max_sequence_gap 9 --slice_length_ft 50 --max_span_length_ft 450; python summarize_v4_realtime_timing.py --timing_csv '$REPLAY/realtime_slice_timing.csv' --output_json '$REPLAY/realtime_timing_summary.json' --output_txt '$REPLAY/realtime_timing_summary.txt'; cat '$REPLAY/realtime_timing_summary.txt'"
echo "Verified replay: $REPLAY_HOST"
