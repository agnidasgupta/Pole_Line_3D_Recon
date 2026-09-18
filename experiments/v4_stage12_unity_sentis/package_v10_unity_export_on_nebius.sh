#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT=$(cat /home/agni/LATEST_V10_UNITY_EXPORT_RUN.txt)
[ -s "$RUN_ROOT/UNITY_EXPORT_COMPLETE.txt" ] || { echo "ERROR: export is not complete" >&2; exit 1; }
STAMP=$(basename "$RUN_ROOT" | sed 's/^unity_export_v10_//')
ARCHIVE=/home/agni/v10_unity_stage12_$STAMP.tar.gz
tar -C "$(dirname "$RUN_ROOT")" -czf "$ARCHIVE" "$(basename "$RUN_ROOT")"
sha256sum "$ARCHIVE" >"$ARCHIVE.sha256"
printf '%s\n' "$ARCHIVE" >/home/agni/LATEST_V10_UNITY_EXPORT_ARCHIVE.txt
echo "ARCHIVE=$ARCHIVE"
cat "$ARCHIVE.sha256"
