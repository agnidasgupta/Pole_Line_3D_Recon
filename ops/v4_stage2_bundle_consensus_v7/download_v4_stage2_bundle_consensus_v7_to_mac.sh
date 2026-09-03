#!/usr/bin/env bash
set -euo pipefail

NEBIUS_HOST="${NEBIUS_HOST:-nebius-va}"
REMOTE_POINTER="${REMOTE_POINTER:-\$HOME/LATEST_V4_STAGE2_BUNDLE_CONSENSUS_V7_ARCHIVE.txt}"
LOCAL_BASE="${LOCAL_BASE:-$HOME/Downloads/v4_stage23_quality}"
LOCAL_PACKAGES="$LOCAL_BASE/packages"

mkdir -p "$LOCAL_PACKAGES"

REMOTE_ARCHIVE=$(
  ssh "$NEBIUS_HOST" "cat \"$REMOTE_POINTER\""
)
[ -n "$REMOTE_ARCHIVE" ] || { echo "ERROR: empty remote archive pointer" >&2; exit 1; }

REMOTE_CHECKSUM="${REMOTE_ARCHIVE}.sha256"
ARCHIVE_NAME=$(basename "$REMOTE_ARCHIVE")
LOCAL_ARCHIVE="$LOCAL_PACKAGES/$ARCHIVE_NAME"
LOCAL_CHECKSUM="${LOCAL_ARCHIVE}.sha256"

echo "Remote archive: $REMOTE_ARCHIVE"
echo "Local archive:  $LOCAL_ARCHIVE"

ssh "$NEBIUS_HOST" "test -s '$REMOTE_ARCHIVE' && test -s '$REMOTE_CHECKSUM' && ls -lh '$REMOTE_ARCHIVE' '$REMOTE_CHECKSUM'"

if command -v rsync >/dev/null 2>&1; then
  rsync -avh --progress --partial "$NEBIUS_HOST:$REMOTE_ARCHIVE" "$LOCAL_PACKAGES/"
  rsync -avh --progress "$NEBIUS_HOST:$REMOTE_CHECKSUM" "$LOCAL_PACKAGES/"
else
  scp "$NEBIUS_HOST:$REMOTE_ARCHIVE" "$LOCAL_PACKAGES/"
  scp "$NEBIUS_HOST:$REMOTE_CHECKSUM" "$LOCAL_PACKAGES/"
fi

EXPECTED=$(awk '{print $1}' "$LOCAL_CHECKSUM")
ACTUAL=$(shasum -a 256 "$LOCAL_ARCHIVE" | awk '{print $1}')
echo "Expected SHA256: $EXPECTED"
echo "Actual SHA256:   $ACTUAL"
[ "$EXPECTED" = "$ACTUAL" ] || { echo "ERROR: SHA256 mismatch" >&2; exit 1; }

tar -tzf "$LOCAL_ARCHIVE" >/dev/null
if tar -tzf "$LOCAL_ARCHIVE" | grep -Ei '\.(npz|pt|pth|ckpt|onnx|engine|safetensors|joblib)$' >/dev/null; then
  echo "ERROR: archive contains an excluded model/cache artifact" >&2
  exit 1
fi

case "$ARCHIVE_NAME" in
  v4_stage2_bundle_consensus_full_run_v2_*.tar.gz)
    RUN_ID=${ARCHIVE_NAME#v4_stage2_bundle_consensus_}
    RUN_ID=${RUN_ID%.tar.gz}
    ;;
  v4_stage2_bundle_consensus_selection_full_run_v2_*.tar.gz)
    RUN_ID=${ARCHIVE_NAME#v4_stage2_bundle_consensus_selection_}
    RUN_ID=${RUN_ID%.tar.gz}
    RUN_ID="${RUN_ID}_selection_diagnostic"
    ;;
  *)
    echo "ERROR: unrecognized archive name: $ARCHIVE_NAME" >&2
    exit 1
    ;;
esac

LOCAL_RUN="$LOCAL_BASE/$RUN_ID"
[ ! -e "$LOCAL_RUN" ] || {
  echo "ERROR: local canonical run directory already exists: $LOCAL_RUN" >&2
  exit 1
}
mkdir -p "$LOCAL_RUN"
tar -xzf "$LOCAL_ARCHIVE" -C "$LOCAL_RUN"

if [ -f "$LOCAL_RUN/STAGE2_ONLY_COMPLETE.txt" ]; then
  [ -f "$LOCAL_RUN/PHASE2_STAGE2_OK.txt" ] || { echo "ERROR: Phase2 marker missing" >&2; exit 1; }
  [ -f "$LOCAL_RUN/session_map.tsv" ] || { echo "ERROR: session map missing" >&2; exit 1; }
  TOTAL=$(wc -l < "$LOCAL_RUN/session_map.tsv" | tr -d ' ')
  MANIFESTS=$(find "$LOCAL_RUN/stage2" -mindepth 2 -maxdepth 2 -type f -name inference_manifest.csv | wc -l | tr -d ' ')
  TIMINGS=$(find "$LOCAL_RUN/timing/stage2" -maxdepth 1 -type f -name '*.csv' | wc -l | tr -d ' ')
  echo "Expected sessions: $TOTAL"
  echo "Stage2 manifests:  $MANIFESTS"
  echo "Timing files:      $TIMINGS"
  [ "$TOTAL" -eq "$MANIFESTS" ] || { echo "ERROR: manifest count mismatch" >&2; exit 1; }
  [ "$TOTAL" -eq "$TIMINGS" ] || { echo "ERROR: timing count mismatch" >&2; exit 1; }
  echo "V4_STAGE2_BUNDLE_CONSENSUS_DOWNLOAD_OK"
else
  [ -f "$LOCAL_RUN/NO_SAFE_BUNDLE_CONSENSUS_PROFILE.txt" ] || {
    echo "ERROR: neither success nor no-safe-profile marker found" >&2
    exit 1
  }
  echo "V4_STAGE2_BUNDLE_CONSENSUS_DIAGNOSTIC_DOWNLOAD_OK"
fi

echo "Archive:       $LOCAL_ARCHIVE"
echo "Extracted run: $LOCAL_RUN"
