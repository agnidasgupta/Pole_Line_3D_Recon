#!/usr/bin/env bash
# Compatibility launcher; implementation: scripts/deploy/v4_production_acceptance_inside_docker.sh
_poleline_script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../scripts/deploy/v4_production_acceptance_inside_docker.sh"
if [[ ! -f "$_poleline_script" ]]; then
  _poleline_script="/workspace/poleline_repo/scripts/deploy/v4_production_acceptance_inside_docker.sh"
fi
source "$_poleline_script"
