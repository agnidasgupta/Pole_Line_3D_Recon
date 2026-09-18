#!/usr/bin/env bash
POLELINE_LEGACY_SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/run_v62_one_session_timing_all_on_nebius.sh"
set -euo pipefail
SESSION_FILTER="${SESSION_FILTER:?set SESSION_FILTER, e.g. 59768101-C4990BB-2026/session3}"
export SESSION_FILTER
./run_v62_one_session_inference_timed_on_nebius.sh
./run_v62_one_session_reconstruction_timed_on_nebius.sh
