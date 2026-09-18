#!/usr/bin/env bash
# Compatibility launcher; implementation: scripts/deploy/show_v4_build_state_on_nebius.sh
_poleline_script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../scripts/deploy/show_v4_build_state_on_nebius.sh"
if [[ ! -f "$_poleline_script" ]]; then
  _poleline_script="/workspace/poleline_repo/scripts/deploy/show_v4_build_state_on_nebius.sh"
fi
source "$_poleline_script"
