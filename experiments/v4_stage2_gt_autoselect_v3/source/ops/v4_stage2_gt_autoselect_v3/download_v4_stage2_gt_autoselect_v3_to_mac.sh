#!/usr/bin/env bash
set -euo pipefail

SSH_ALIAS="${SSH_ALIAS:-nebius-va}"
LOCAL_ROOT="${LOCAL_ROOT:-$HOME/Downloads/v4_stage2_gt_autoselect_v3}"
REMOTE_ARCHIVE=$(ssh "$SSH_ALIAS" 'cat "$HOME/LATEST_V4_STAGE2_GT_AUTOSELECT_V3_ARCHIVE.txt"')
[ -n "$REMOTE_ARCHIVE" ] || { echo "ERROR: remote archive pointer is empty" >&2; exit 1; }
mkdir -p "$LOCAL_ROOT"
scp "$SSH_ALIAS:$REMOTE_ARCHIVE" "$LOCAL_ROOT/"
scp "$SSH_ALIAS:${REMOTE_ARCHIVE}.sha256" "$LOCAL_ROOT/"
LOCAL_ARCHIVE="$LOCAL_ROOT/$(basename "$REMOTE_ARCHIVE")"
LOCAL_SHA="$LOCAL_ROOT/$(basename "$REMOTE_ARCHIVE").sha256"
EXPECTED=$(awk '{print $1}' "$LOCAL_SHA")
ACTUAL=$(shasum -a 256 "$LOCAL_ARCHIVE" | awk '{print $1}')
echo "Expected SHA256: $EXPECTED"
echo "Actual SHA256:   $ACTUAL"
[ "$EXPECTED" = "$ACTUAL" ] || { echo "ERROR: SHA256 mismatch" >&2; exit 2; }
tar -tzf "$LOCAL_ARCHIVE" >/dev/null
EXTRACT_DIR="$LOCAL_ROOT/extracted/$(basename "$LOCAL_ARCHIVE" .tar.gz)"
mkdir -p "$EXTRACT_DIR"
tar -xzf "$LOCAL_ARCHIVE" -C "$EXTRACT_DIR"
echo "STAGE2_GT_AUTOSELECT_DOWNLOAD_OK"
echo "Archive:   $LOCAL_ARCHIVE"
echo "Extracted: $EXTRACT_DIR"
