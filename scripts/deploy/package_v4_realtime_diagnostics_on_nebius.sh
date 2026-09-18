#!/usr/bin/env bash
POLELINE_LEGACY_SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/v4/package_v4_realtime_diagnostics_on_nebius.sh"
set -euo pipefail
HERE=$(cd "$(dirname "${POLELINE_LEGACY_SOURCE}")" && pwd)
exec "$HERE/package_v4_review_bundle_on_nebius.sh"
