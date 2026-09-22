#!/usr/bin/env bash
POLELINE_LEGACY_SOURCE="${BASH_SOURCE[0]}"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${POLELINE_LEGACY_SOURCE}")" && pwd)"
exec "$SCRIPT_DIR/download_v62_teacher_results_to_mac.sh" "$@"
