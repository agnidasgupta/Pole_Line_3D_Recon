#!/usr/bin/env bash
# Compatibility launcher; implementation: scripts/deploy/package_v4_production_state_on_nebius.sh
_poleline_script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../scripts/deploy/package_v4_production_state_on_nebius.sh"
if [[ ! -f "$_poleline_script" ]]; then
  _poleline_script="/workspace/poleline_repo/scripts/deploy/package_v4_production_state_on_nebius.sh"
fi
source "$_poleline_script"
