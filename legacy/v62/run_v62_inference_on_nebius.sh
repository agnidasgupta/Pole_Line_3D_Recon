#!/usr/bin/env bash
POLELINE_LEGACY_SOURCE="${BASH_SOURCE[0]}"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
set -euo pipefail
exec ./run_v62_teacher_inference_on_nebius.sh "$@"
