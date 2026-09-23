#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Runtime gate now includes batch-size tuning so runtime mode and batch shape remain one validated decision.
exec "$HERE/run_v4_runtime_variant_gate_on_nebius.sh"
