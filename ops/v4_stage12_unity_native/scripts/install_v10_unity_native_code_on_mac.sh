#!/usr/bin/env bash
set -euo pipefail

: "${UNITY_PROJECT:?Set UNITY_PROJECT to the Unity project directory}"
SCRIPT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$UNITY_PROJECT/Assets/Scripts/V10Stage12" "$UNITY_PROJECT/Assets/Editor" "$UNITY_PROJECT/Assets/V10Stage12/Models"
rsync -a --delete "$SCRIPT_ROOT/Assets/Scripts/V10Stage12/" "$UNITY_PROJECT/Assets/Scripts/V10Stage12/"
rsync -a "$SCRIPT_ROOT/Assets/Editor/" "$UNITY_PROJECT/Assets/Editor/"
echo "UNITY_CODE_INSTALLED=$UNITY_PROJECT"
echo "Next: extract the generated model-assets archive at the Unity project root."
