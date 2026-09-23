#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
{
  while IFS= read -r f; do
    (cd "$HERE" && sha256sum "$f")
  done < <(find "$HERE" -maxdepth 1 -type f \( -name '*.py' -o -name '*.sh' -o -name 'requirements.txt' -o -name 'Dockerfile.v4_realtime' \) -printf '%f\n' | sort)
} | sha256sum | awk '{print $1}'
