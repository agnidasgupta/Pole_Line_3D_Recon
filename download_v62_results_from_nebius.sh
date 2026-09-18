#!/usr/bin/env bash
# Compatibility launcher; implementation: legacy/v62/download_v62_results_from_nebius.sh
_poleline_script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/legacy/v62/download_v62_results_from_nebius.sh"
if [[ ! -f "$_poleline_script" ]]; then
  _poleline_script="/workspace/poleline_repo/legacy/v62/download_v62_results_from_nebius.sh"
fi
source "$_poleline_script"
