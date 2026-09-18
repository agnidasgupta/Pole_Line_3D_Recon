#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
V4_ALL_MODE=reconstruct exec "$HERE/run_v4_all_data_mode_on_nebius.sh"
