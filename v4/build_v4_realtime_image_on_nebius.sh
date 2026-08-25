#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IMAGE=${IMAGE:-va-v4-realtime:torch241-cu121}
if docker info >/dev/null 2>&1; then DOCKER=(docker); else DOCKER=(sudo docker); fi
"${DOCKER[@]}" build -f "$HERE/Dockerfile.v4_realtime" -t "$IMAGE" "$HERE"
echo "Built $IMAGE"
"${DOCKER[@]}" run --rm --gpus all "$IMAGE" python - <<'PY'
import torch, scipy, sklearn, pandas, numpy
print('torch=',torch.__version__,'cuda=',torch.version.cuda,'cuda_available=',torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu=',torch.cuda.get_device_name(0),'bf16=',torch.cuda.is_bf16_supported())
print('scipy=',scipy.__version__,'sklearn=',sklearn.__version__,'pandas=',pandas.__version__,'numpy=',numpy.__version__)
PY
