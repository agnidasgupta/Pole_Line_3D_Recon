#!/usr/bin/env bash
# Compatibility launcher; implementation: scripts/deploy/run_v4_stage2_training_on_nebius.sh
_poleline_script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../scripts/deploy/run_v4_stage2_training_on_nebius.sh"
if [[ ! -f "$_poleline_script" ]]; then
  _poleline_script="/workspace/poleline_repo/scripts/deploy/run_v4_stage2_training_on_nebius.sh"
fi
source "$_poleline_script"
