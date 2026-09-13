#!/usr/bin/env bash
set -euo pipefail

: "${UNITY_PROJECT:?Set UNITY_PROJECT to the Unity project directory}"
: "${UNITY_EDITOR:?Set UNITY_EDITOR to the Unity 6.3 editor executable}"

export V10_UNITY_BUILD_KIND="${V10_UNITY_BUILD_KIND:-server}"
default_name=V10Stage12Server
[[ "$V10_UNITY_BUILD_KIND" == player ]] && default_name=V10Stage12GpuPlayer
export V10_UNITY_BUILD_OUTPUT="${V10_UNITY_BUILD_OUTPUT:-$UNITY_PROJECT/Builds/Linux/$default_name}"
LOG="${V10_UNITY_BUILD_LOG:-$UNITY_PROJECT/Builds/Linux/V10Stage12Build.log}"
mkdir -p "$(dirname "$LOG")"

"$UNITY_EDITOR" -batchmode -nographics -quit \
  -projectPath "$UNITY_PROJECT" \
  -executeMethod V10Stage12Build.BuildLinuxServer \
  -logFile "$LOG"

test -x "$V10_UNITY_BUILD_OUTPUT"
grep -q 'V10_STAGE12_NATIVE_BUILD_OK' "$LOG"
echo "BUILD_OK=$V10_UNITY_BUILD_OUTPUT"
echo "LOG=$LOG"
