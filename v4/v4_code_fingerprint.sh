#!/usr/bin/env bash
# Compatibility launcher; implementation: scripts/deploy/v4_code_fingerprint.sh
_poleline_script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../scripts/deploy/v4_code_fingerprint.sh"
if [[ ! -f "$_poleline_script" ]]; then
  _poleline_script="/workspace/poleline_repo/scripts/deploy/v4_code_fingerprint.sh"
fi
source "$_poleline_script"
