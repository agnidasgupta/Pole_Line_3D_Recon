#!/usr/bin/env bash
# Run on the local Mac. ssh/scp/shasum only; no Python.
set -euo pipefail
REMOTE_HOST=${REMOTE_HOST:-nebius-va}
LOCAL_ROOT=${LOCAL_ROOT:-$HOME/Downloads/v4_all_data_results}
mkdir -p "$LOCAL_ROOT"

ARCH=$(ssh "$REMOTE_HOST" '
  set -e
  ptr=$(for r in "$HOME" /workspace/voxel_poleline/outputs /outputs /data/outputs; do
    [[ -d "$r" ]] || continue
    find "$r" -type f -path "*/v4_production/full_dataset_packages/LATEST_ALL_DATA_ARCHIVE.txt" -printf "%T@ %p\n" 2>/dev/null
  done | sort -nr | head -1 | cut -d" " -f2-)
  arch=""
  if [[ -n "$ptr" && -s "$ptr" ]]; then arch=$(cat "$ptr"); fi
  if [[ -z "$arch" || ! -s "$arch" ]]; then
    arch=$(for r in "$HOME" /workspace/voxel_poleline/outputs /outputs /data/outputs; do
      [[ -d "$r" ]] || continue
      find "$r" -type f -name "v4_all_data_results_*.tar.gz" -printf "%T@ %p\n" 2>/dev/null
    done | sort -nr | head -1 | cut -d" " -f2-)
  fi
  [[ -n "$arch" && -s "$arch" ]] || exit 7
  printf "%s\n" "$arch"
' 2>/dev/null || true)
[[ -n "$ARCH" ]] || { echo "ERROR: no V4 full-dataset results archive found on $REMOTE_HOST" >&2; exit 2; }
BASE=$(basename "$ARCH")
scp "$REMOTE_HOST:$ARCH" "$LOCAL_ROOT/$BASE"
EXPECTED=$(ssh "$REMOTE_HOST" "sha256sum '$ARCH' | awk '{print \$1}'")
ACTUAL=$(shasum -a 256 "$LOCAL_ROOT/$BASE" | awk '{print $1}')
[[ -n "$EXPECTED" && "$EXPECTED" == "$ACTUAL" ]] || { echo "ERROR: SHA256 mismatch expected=$EXPECTED actual=$ACTUAL" >&2; exit 3; }
printf '%s  %s\n' "$ACTUAL" "$BASE" > "$LOCAL_ROOT/$BASE.sha256"
echo "SHA256 verified: $ACTUAL"
echo "Downloaded: $LOCAL_ROOT/$BASE"
echo "The archive excludes .npz and model/checkpoint files by construction."
