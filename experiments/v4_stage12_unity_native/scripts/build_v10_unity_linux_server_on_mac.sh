#!/usr/bin/env bash
set -euo pipefail

: "${UNITY_PROJECT:?Set UNITY_PROJECT to the Unity project directory}"
: "${UNITY_EDITOR:?Set UNITY_EDITOR to the Unity 6.3 editor executable}"

export V10_UNITY_BUILD_KIND="${V10_UNITY_BUILD_KIND:-gpu-player}"
case "$V10_UNITY_BUILD_KIND" in
  gpu-player)
    default_name=V10Stage12GpuPlayer
    build_method=V10Stage12Build.BuildLinuxGpuPlayer
    ;;
  cpu-server)
    default_name=V10Stage12CpuServer
    build_method=V10Stage12Build.BuildLinuxCpuServer
    ;;
  *)
    echo "ERROR: V10_UNITY_BUILD_KIND must be gpu-player or cpu-server" >&2
    exit 1
    ;;
esac
export V10_UNITY_BUILD_OUTPUT="${V10_UNITY_BUILD_OUTPUT:-$UNITY_PROJECT/Builds/Linux/$default_name}"
LOG="${V10_UNITY_BUILD_LOG:-$UNITY_PROJECT/Builds/Linux/V10Stage12Build.log}"
mkdir -p "$(dirname "$LOG")"

grep -q '"com.unity.ai.inference"[[:space:]]*:[[:space:]]*"2.6.1"' "$UNITY_PROJECT/Packages/manifest.json" || {
  echo "ERROR: com.unity.ai.inference 2.6.1 is not pinned in Packages/manifest.json" >&2
  echo "Run install_v10_unity_native_code_on_mac.sh again." >&2
  exit 1
}

set +e
"$UNITY_EDITOR" -batchmode -nographics -quit \
  -projectPath "$UNITY_PROJECT" \
  -executeMethod "$build_method" \
  -logFile "$LOG"
rc=$?
set -e

if [[ "$rc" -ne 0 ]]; then
  echo "UNITY_BUILD_FAILED exit_code=$rc" >&2
  grep -nE 'error CS[0-9]+|Scripts have compiler errors|BuildFailedException|Exception:' "$LOG" | tail -n 200 >&2 || true
  echo "FULL_LOG=$LOG" >&2
  exit "$rc"
fi

test -x "$V10_UNITY_BUILD_OUTPUT"
grep -q 'V10_STAGE12_NATIVE_BUILD_OK' "$LOG"
echo "BUILD_OK=$V10_UNITY_BUILD_OUTPUT"
echo "BUILD_KIND=$V10_UNITY_BUILD_KIND"
echo "LOG=$LOG"
