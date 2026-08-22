#!/usr/bin/env bash
set -euo pipefail
SESSION_FILTER="${SESSION_FILTER:?set SESSION_FILTER, e.g. 59768101-C4990BB-2026/session3}"
export SESSION_FILTER
./run_v62_one_session_inference_timed_on_nebius.sh
./run_v62_one_session_reconstruction_timed_on_nebius.sh
