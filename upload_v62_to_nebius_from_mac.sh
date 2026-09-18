#!/usr/bin/env bash
# Compatibility launcher; implementation: legacy/v62/upload_v62_to_nebius_from_mac.sh
_poleline_script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/legacy/v62/upload_v62_to_nebius_from_mac.sh"
if [[ ! -f "$_poleline_script" ]]; then
  _poleline_script="/workspace/poleline_repo/legacy/v62/upload_v62_to_nebius_from_mac.sh"
fi
source "$_poleline_script"
