#!/usr/bin/env bash
POLELINE_LEGACY_SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/download_v62_results_from_nebius.sh"
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${POLELINE_LEGACY_SOURCE}")" && pwd)"
exec "$SCRIPT_DIR/download_v62_teacher_results_to_mac.sh" "$@"
