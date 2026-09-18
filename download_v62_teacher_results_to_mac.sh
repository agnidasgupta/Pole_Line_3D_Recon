#!/usr/bin/env bash
# Compatibility launcher; implementation: legacy/v62/download_v62_teacher_results_to_mac.sh
_poleline_script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/legacy/v62/download_v62_teacher_results_to_mac.sh"
if [[ ! -f "$_poleline_script" ]]; then
  _poleline_script="/workspace/poleline_repo/legacy/v62/download_v62_teacher_results_to_mac.sh"
fi
source "$_poleline_script"
