#!/usr/bin/env bash
POLELINE_LEGACY_SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/v4/run_v4_stage1_batch_sweep_on_nebius.sh"
set -euo pipefail
HERE=$(cd "$(dirname "${POLELINE_LEGACY_SOURCE}")" && pwd)
# Runtime gate now includes batch-size tuning so runtime mode and batch shape remain one validated decision.
exec "$HERE/run_v4_runtime_variant_gate_on_nebius.sh"
