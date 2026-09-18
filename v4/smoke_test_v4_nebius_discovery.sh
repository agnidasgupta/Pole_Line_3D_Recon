#!/usr/bin/env bash
# Compatibility launcher; implementation: scripts/deploy/smoke_test_v4_nebius_discovery.sh
_poleline_script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../scripts/deploy/smoke_test_v4_nebius_discovery.sh"
if [[ ! -f "$_poleline_script" ]]; then
  _poleline_script="/workspace/poleline_repo/scripts/deploy/smoke_test_v4_nebius_discovery.sh"
fi
source "$_poleline_script"
