#!/usr/bin/env bash
set -euo pipefail
TOOL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_IMAGE=${BASE_IMAGE:-va-v4-realtime:torch241-cu121}
PROFILE_IMAGE=${PROFILE_IMAGE:-va-v4-realtime:torch241-cu121-nsight}
docker build \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  -f "$TOOL_DIR/Dockerfile.nsight-systems" \
  -t "$PROFILE_IMAGE" "$TOOL_DIR"
docker run --rm "$PROFILE_IMAGE" nsys --version
echo "NSIGHT_IMAGE_OK image=$PROFILE_IMAGE"
