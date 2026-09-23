#!/usr/bin/env bash
# Pure host-shell packaging: no Python dependency.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/v4_nebius_common.sh"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
PKG="$V4_SESSION_ROOT_HOST/backups/v4_state_${V4_SESSION_SAFE}_${V4_DEPLOY_SHORT}_${STAMP}.tar.gz"
LIST=$(mktemp)
trap 'rm -f "$LIST"' EXIT
printf '%s\n' "run_context.env" "LATEST_RUN_ROOT.txt" >> "$LIST"
# Store paths relative to the session root when present. Include diagnostics and the active deployment run.
REL_RUN=${V4_RUN_ROOT_HOST#"$V4_SESSION_ROOT_HOST"/}
[[ "$REL_RUN" != "$V4_RUN_ROOT_HOST" ]] && printf '%s\n' "$REL_RUN" >> "$LIST"
[[ -d "$V4_DIAG_ROOT_HOST" ]] && {
  # Diagnostics may be outside the session root; package separately below when needed.
  :
}
printf '%s\n' "backups/LATEST_CODE_BACKUP.txt" >> "$LIST"
# First archive the session-local durable state without nested backup archives.
tar -C "$V4_SESSION_ROOT_HOST" -czf "$PKG" --exclude='backups/*.tar.gz' -T "$LIST" 2>/dev/null || \
  tar -C "$V4_SESSION_ROOT_HOST" -czf "$PKG" --exclude='backups/*.tar.gz' .
# If diagnostics are outside the session root, create a sibling diagnostics snapshot as a second durable artifact.
DIAG_PKG="${PKG%.tar.gz}_diagnostics.tar.gz"
if [[ -d "$V4_DIAG_ROOT_HOST" ]]; then
  tar -C "$V4_DIAG_ROOT_HOST" -czf "$DIAG_PKG" .
  echo "Diagnostics package: $DIAG_PKG"
fi
echo "State package: $PKG"
