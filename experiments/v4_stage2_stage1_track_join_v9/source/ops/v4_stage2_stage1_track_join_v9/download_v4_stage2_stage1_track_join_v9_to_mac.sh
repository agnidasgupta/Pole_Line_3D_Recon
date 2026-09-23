#!/usr/bin/env bash
set -euo pipefail
REMOTE="${REMOTE:-nebius-va}"
REMOTE_ARCHIVE=$(ssh "$REMOTE" 'cat "$HOME/LATEST_V4_STAGE2_STAGE1_TRACK_JOIN_V9_ARCHIVE.txt"')
REMOTE_RUN=$(ssh "$REMOTE" 'cat "$HOME/LATEST_V4_STAGE2_STAGE1_TRACK_JOIN_V9_RUN.txt"')
[ -n "$REMOTE_ARCHIVE" ] || { echo "ERROR: archive pointer empty" >&2; exit 1; }
[ -n "$REMOTE_RUN" ] || { echo "ERROR: run pointer empty" >&2; exit 1; }
RUN_ID=$(basename "$REMOTE_RUN")
[[ "$RUN_ID" =~ ^full_run_v2_[0-9]{8}T[0-9]{6}Z$ ]] || { echo "ERROR: unexpected run id: $RUN_ID" >&2; exit 1; }
LOCAL_BASE="$HOME/Downloads/v4_stage23_quality"
LOCAL_PACKAGES="$LOCAL_BASE/packages"
LOCAL_RUN="$LOCAL_BASE/$RUN_ID"
mkdir -p "$LOCAL_PACKAGES"
LOCAL_ARCHIVE="$LOCAL_PACKAGES/$(basename "$REMOTE_ARCHIVE")"
rsync -avh --progress --partial "$REMOTE:$REMOTE_ARCHIVE" "$LOCAL_PACKAGES/"
rsync -avh --progress "$REMOTE:${REMOTE_ARCHIVE}.sha256" "$LOCAL_PACKAGES/"
EXPECTED=$(awk '{print $1}' "${LOCAL_ARCHIVE}.sha256")
ACTUAL=$(shasum -a 256 "$LOCAL_ARCHIVE" | awk '{print $1}')
echo "Expected: $EXPECTED"
echo "Actual:   $ACTUAL"
[ "$EXPECTED" = "$ACTUAL" ] || { echo "ERROR: checksum mismatch" >&2; exit 1; }
tar -tzf "$LOCAL_ARCHIVE" >/dev/null
if tar -tzf "$LOCAL_ARCHIVE" | grep -Ei '\.(npz|pt|pth|ckpt|onnx|engine|safetensors|joblib)$' >/dev/null; then
  echo "ERROR: excluded artifact in archive" >&2; exit 1
fi
if [ -e "$LOCAL_RUN" ]; then
  OLD="${LOCAL_RUN}.previous_$(date -u +%Y%m%dT%H%M%SZ)"
  mv "$LOCAL_RUN" "$OLD"
  echo "Preserved previous local run: $OLD"
fi
mkdir -p "$LOCAL_RUN"
tar -xzf "$LOCAL_ARCHIVE" -C "$LOCAL_RUN"
echo "V9 STAGE1 TRACK-JOIN DOWNLOAD OK"
echo "Local run: $LOCAL_RUN"
echo "Stage2:   $LOCAL_RUN/stage2"
