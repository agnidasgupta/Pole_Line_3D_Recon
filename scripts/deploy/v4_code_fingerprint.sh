#!/usr/bin/env bash
POLELINE_LEGACY_SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/v4/v4_code_fingerprint.sh"
set -euo pipefail
HERE=$(cd "$(dirname "${POLELINE_LEGACY_SOURCE}")" && pwd)
python3 "$HERE/../scripts/validation/fingerprint.py"
