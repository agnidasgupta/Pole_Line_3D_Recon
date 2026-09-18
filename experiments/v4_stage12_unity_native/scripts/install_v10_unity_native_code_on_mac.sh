#!/usr/bin/env bash
set -euo pipefail

: "${UNITY_PROJECT:?Set UNITY_PROJECT to the Unity project directory}"
SCRIPT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
MANIFEST="$UNITY_PROJECT/Packages/manifest.json"

test -f "$MANIFEST" || {
  echo "ERROR: Unity package manifest not found: $MANIFEST" >&2
  exit 1
}

# Pin the exact API used by this source. This uses only Python's standard
# library; model inference and reconstruction never use host Python.
python3 - "$MANIFEST" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text())
dependencies = data.setdefault("dependencies", {})
dependencies["com.unity.ai.inference"] = "2.6.1"
data["dependencies"] = dict(sorted(dependencies.items()))
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(data, indent=2) + "\n")
temporary.replace(path)
PY

mkdir -p "$UNITY_PROJECT/Assets/Scripts/V10Stage12" "$UNITY_PROJECT/Assets/Editor" "$UNITY_PROJECT/Assets/V10Stage12/Models"
rsync -a --delete "$SCRIPT_ROOT/Assets/Scripts/V10Stage12/" "$UNITY_PROJECT/Assets/Scripts/V10Stage12/"
rsync -a "$SCRIPT_ROOT/Assets/Editor/" "$UNITY_PROJECT/Assets/Editor/"
echo "UNITY_CODE_INSTALLED=$UNITY_PROJECT"
echo "SENTIS_PACKAGE_PINNED=com.unity.ai.inference@2.6.1"
echo "Next: extract the generated model-assets archive at the Unity project root."
