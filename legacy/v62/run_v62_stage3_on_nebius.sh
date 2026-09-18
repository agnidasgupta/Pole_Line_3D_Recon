#!/usr/bin/env bash
POLELINE_LEGACY_SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/run_v62_stage3_on_nebius.sh"
set -euo pipefail
exec ./run_v62_teacher_reconstruction_on_nebius.sh "$@"
