#!/usr/bin/env bash
# Compatibility launcher; implementation: scripts/deploy/verify_v4_realtime_replay_on_nebius.sh
_poleline_script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../scripts/deploy/verify_v4_realtime_replay_on_nebius.sh"
if [[ ! -f "$_poleline_script" ]]; then
  _poleline_script="/workspace/poleline_repo/scripts/deploy/verify_v4_realtime_replay_on_nebius.sh"
fi
source "$_poleline_script"
