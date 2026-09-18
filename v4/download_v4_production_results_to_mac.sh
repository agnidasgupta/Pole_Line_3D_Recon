#!/usr/bin/env bash
# Compatibility launcher; implementation: scripts/deploy/download_v4_production_results_to_mac.sh
_poleline_script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../scripts/deploy/download_v4_production_results_to_mac.sh"
if [[ ! -f "$_poleline_script" ]]; then
  _poleline_script="/workspace/poleline_repo/scripts/deploy/download_v4_production_results_to_mac.sh"
fi
source "$_poleline_script"
