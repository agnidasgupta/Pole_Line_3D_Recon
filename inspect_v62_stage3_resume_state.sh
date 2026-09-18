#!/usr/bin/env bash
# Compatibility launcher; implementation: legacy/v62/inspect_v62_stage3_resume_state.sh
_poleline_script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/legacy/v62/inspect_v62_stage3_resume_state.sh"
if [[ ! -f "$_poleline_script" ]]; then
  _poleline_script="/workspace/poleline_repo/legacy/v62/inspect_v62_stage3_resume_state.sh"
fi
source "$_poleline_script"
