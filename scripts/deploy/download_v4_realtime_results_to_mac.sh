#!/usr/bin/env bash
POLELINE_LEGACY_SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/v4/download_v4_realtime_results_to_mac.sh"
set -euo pipefail
HERE=$(cd "$(dirname "${POLELINE_LEGACY_SOURCE}")" && pwd)
exec "$HERE/download_v4_review_bundle_to_mac.sh"
