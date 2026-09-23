#!/usr/bin/env bash
# Run on the local Mac. Shell/ssh/scp/shasum only; no Python.
set -euo pipefail
REMOTE_HOST=${REMOTE_HOST:-nebius-va}
LOCAL_ROOT=${LOCAL_ROOT:-$HOME/Downloads/v4_production_review}
mkdir -p "$LOCAL_ROOT"
LATEST=$(ssh "$REMOTE_HOST" 'for r in /workspace/voxel_poleline/outputs /outputs /data/outputs; do [[ -d "$r" ]] || continue; find "$r" -type f -path "*/v4_production/review_packages/LATEST_REVIEW_ARCHIVE.txt" -printf "%T@ %p\n" 2>/dev/null; done | sort -nr | head -1 | cut -d" " -f2-')
[[ -n "$LATEST" ]] || { echo "ERROR: no V4 review archive pointer found on $REMOTE_HOST" >&2; exit 2; }
ARCHIVE=$(ssh "$REMOTE_HOST" "cat '$LATEST'")
[[ -n "$ARCHIVE" ]] || { echo "ERROR: empty remote archive pointer: $LATEST" >&2; exit 2; }
BASE=$(basename "$ARCHIVE")
scp "$REMOTE_HOST:$ARCHIVE" "$LOCAL_ROOT/$BASE"
scp "$REMOTE_HOST:$ARCHIVE.sha256" "$LOCAL_ROOT/$BASE.sha256.remote"
EXPECTED=$(awk 'NR==1{print $1}' "$LOCAL_ROOT/$BASE.sha256.remote")
ACTUAL=$(shasum -a 256 "$LOCAL_ROOT/$BASE" | awk '{print $1}')
[[ -n "$EXPECTED" && "$EXPECTED" == "$ACTUAL" ]] || { echo "ERROR: SHA256 mismatch expected=$EXPECTED actual=$ACTUAL" >&2; exit 3; }
printf '%s  %s\n' "$ACTUAL" "$BASE" > "$LOCAL_ROOT/$BASE.sha256"
rm -f "$LOCAL_ROOT/$BASE.sha256.remote"
echo "SHA256 verified: $ACTUAL"
echo "Downloaded: $LOCAL_ROOT/$BASE"
echo "Upload that .tar.gz file to ChatGPT for review."
