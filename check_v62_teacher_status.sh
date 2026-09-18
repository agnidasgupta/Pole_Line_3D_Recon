#!/usr/bin/env bash
# Compatibility launcher; implementation: legacy/v62/check_v62_teacher_status.sh
_poleline_script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/legacy/v62/check_v62_teacher_status.sh"
if [[ ! -f "$_poleline_script" ]]; then
  _poleline_script="/workspace/poleline_repo/legacy/v62/check_v62_teacher_status.sh"
fi
source "$_poleline_script"
