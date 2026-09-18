#!/usr/bin/env bash
# Run full-dataset Stage 3 only from the latest durable Stage 2 results.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
V4_ALL_MODE=stage3 exec "$HERE/run_v4_all_data_mode_on_nebius.sh"
