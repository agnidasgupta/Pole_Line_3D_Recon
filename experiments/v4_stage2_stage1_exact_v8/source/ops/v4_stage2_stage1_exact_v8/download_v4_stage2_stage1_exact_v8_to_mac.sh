#!/usr/bin/env bash
set -euo pipefail
REMOTE="${REMOTE:-nebius-va}"
LOCAL_BASE="${LOCAL_BASE:-$HOME/Downloads/v4_stage23_quality}"
mkdir -p "$LOCAL_BASE/packages"
REMOTE_ARCHIVE=$(ssh "$REMOTE" 'cat "$HOME/LATEST_V4_STAGE2_STAGE1_EXACT_V8_ARCHIVE.txt"')
REMOTE_RUN=$(ssh "$REMOTE" 'cat "$HOME/LATEST_V4_STAGE2_STAGE1_EXACT_V8_RUN.txt"')
RUN_ID=$(basename "$REMOTE_RUN")
LOCAL_ARCHIVE="$LOCAL_BASE/packages/$(basename "$REMOTE_ARCHIVE")"
LOCAL_RUN="$LOCAL_BASE/$RUN_ID"
rsync -avh --progress --partial "$REMOTE:$REMOTE_ARCHIVE" "$LOCAL_BASE/packages/"
rsync -avh --progress "$REMOTE:${REMOTE_ARCHIVE}.sha256" "$LOCAL_BASE/packages/"
EXPECTED=$(awk '{print $1}' "${LOCAL_ARCHIVE}.sha256")
ACTUAL=$(shasum -a 256 "$LOCAL_ARCHIVE" | awk '{print $1}')
test "$EXPECTED" = "$ACTUAL" || { echo "ERROR: SHA256 mismatch"; exit 1; }
tar -tzf "$LOCAL_ARCHIVE" >/dev/null
if tar -tzf "$LOCAL_ARCHIVE" | grep -Ei '\.(npz|joblib|pt|pth|ckpt)$' >/dev/null; then
  echo "ERROR: excluded model/cache file found in archive"; exit 1
fi
test ! -e "$LOCAL_RUN" || { echo "ERROR: local run path exists: $LOCAL_RUN"; exit 1; }
mkdir -p "$LOCAL_RUN"
tar -xzf "$LOCAL_ARCHIVE" -C "$LOCAL_RUN"
cat "$LOCAL_RUN/PHASE2_STAGE2_OK.txt"
cat "$LOCAL_RUN/STAGE2_ONLY_COMPLETE.txt"
echo "V4_STAGE2_STAGE1_EXACT_DOWNLOAD_OK"
echo "local_run=$LOCAL_RUN"
