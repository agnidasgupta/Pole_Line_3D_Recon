#!/usr/bin/env bash
# Host-side helper for the full-dataset V4 operations add-on.
# It never runs Python on the Nebius host.
set -euo pipefail
OPS_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

find_accepted_v4_dir() {
  local c=''
  # Prefer the established Nebius workspace, then Git repositories under /workspace and HOME.
  for candidate in \
    /workspace/voxel_poleline/v4 \
    /workspace/Pole_Line_3D_Recon/v4 \
    "$HOME/Pole_Line_3D_Recon/v4"; do
    if [[ -f "$candidate/v4_nebius_common.sh" && -f "$candidate/v4_code_fingerprint.sh" ]]; then
      (cd "$candidate" && pwd); return 0
    fi
  done
  c=$(find /workspace "$HOME" -maxdepth 6 -type f -name v4_nebius_common.sh -path '*/v4/*' -printf '%T@ %h\n' 2>/dev/null \
      | sort -nr | head -1 | cut -d' ' -f2- || true)
  [[ -n "$c" && -f "$c/v4_code_fingerprint.sh" ]] || return 1
  (cd "$c" && pwd)
}

V4_ACCEPTED_DIR=$(find_accepted_v4_dir || true)
[[ -n "$V4_ACCEPTED_DIR" ]] || { echo "ERROR: accepted V4 code directory could not be auto-discovered." >&2; exit 2; }
# shellcheck disable=SC1090
source "$V4_ACCEPTED_DIR/v4_nebius_common.sh"
