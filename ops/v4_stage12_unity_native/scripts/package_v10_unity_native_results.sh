#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=$(cat /home/agni/LATEST_V10_UNITY_NATIVE_RUN.txt)
test -f "$RUN_ROOT/STAGE12_COMPLETE.txt"
if [[ -f "$RUN_ROOT/UNITY_NATIVE_EQUIVALENCE.txt" ]]; then
  grep -q '^status=PASS$' "$RUN_ROOT/UNITY_NATIVE_EQUIVALENCE.txt"
fi
if find "$RUN_ROOT/status" -maxdepth 1 -type f -name '*.failed' -print -quit | grep -q .; then
  echo "Refusing to package a run with failed status markers" >&2
  exit 1
fi

parent=$(dirname "$RUN_ROOT")
name=$(basename "$RUN_ROOT")
archive="/home/agni/${name}.tar.gz"
tar -C "$parent" -czf "$archive" "$name"
sha256sum "$archive" > "$archive.sha256"
printf '%s\n' "$archive" > /home/agni/LATEST_V10_UNITY_NATIVE_ARCHIVE.txt
echo "PACKAGE_OK=$archive"
cat "$archive.sha256"
