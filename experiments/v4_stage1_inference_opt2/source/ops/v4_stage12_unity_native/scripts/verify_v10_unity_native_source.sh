#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
for script in "$ROOT"/scripts/*.sh; do bash -n "$script"; done
python3 - "$ROOT/tools" <<'PY'
import ast
import pathlib
import sys
for path in pathlib.Path(sys.argv[1]).glob("*.py"):
    ast.parse(path.read_text(), filename=str(path))
PY
if grep -REn 'Process[.]Start|run_unity_stage1_csv_stage2|import (numpy|pandas|joblib|sklearn)' "$ROOT/Assets"; then
  echo "Forbidden runtime dependency found" >&2
  exit 1
fi
grep -q 'disconnected_fragment_bridge_allowed = false' "$ROOT/Assets/Scripts/V10Stage12/V10Stage12Contracts.cs"
grep -q 'geometry_stage1_voxel_support_fraction' "$ROOT/Assets/Scripts/V10Stage12/V10NativeEquivalenceValidator.cs"
grep -q 'stage1_to_stage2_voxel_preservation' "$ROOT/Assets/Scripts/V10Stage12/V10NativeEquivalenceValidator.cs"
grep -q 'com.unity.ai.inference.*2.6.1' "$ROOT/scripts/install_v10_unity_native_code_on_mac.sh"
grep -q 'BuildLinuxGpuPlayer' "$ROOT/Assets/Editor/V10Stage12Build.cs"
grep -q '^using UnityEditor.Build;$' "$ROOT/Assets/Editor/V10Stage12Build.cs"
grep -q '^using UnityEditor.Build.Reporting;$' "$ROOT/Assets/Editor/V10Stage12Build.cs"
grep -q 'BackendType.GPUCompute' "$ROOT/Assets/Editor/V10Stage12Build.cs"
grep -q -- '-force-vulkan' "$ROOT/scripts/launch_v10_unity_native_experiment.sh"
grep -q -- '--v10-input-csv' "$ROOT/Assets/Scripts/V10Stage12/V10NativeSessionRunner.cs"
grep -q -- '--v10-input-directory' "$ROOT/Assets/Scripts/V10Stage12/V10NativeSessionRunner.cs"
grep -q -- '-force-vulkan' "$ROOT/scripts/launch_v10_unity_realtime_handoff.sh"
grep -q 'STALLED' "$ROOT/scripts/monitor_v10_unity_realtime_handoff.sh"
grep -q 'CHECKSUMS.sha256' "$ROOT/scripts/package_v10_unity_realtime_handoff_on_mac.sh"
test -s "$ROOT/REALTIME_HANDOFF_README.md"
echo V10_UNITY_NATIVE_SOURCE_CHECK_OK
