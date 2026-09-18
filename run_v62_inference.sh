#!/usr/bin/env bash
# Compatibility launcher; implementation: legacy/v62/run_v62_inference.sh
_poleline_script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/legacy/v62/run_v62_inference.sh"
if [[ ! -f "$_poleline_script" ]]; then
  _poleline_script="/workspace/poleline_repo/legacy/v62/run_v62_inference.sh"
fi
source "$_poleline_script"
