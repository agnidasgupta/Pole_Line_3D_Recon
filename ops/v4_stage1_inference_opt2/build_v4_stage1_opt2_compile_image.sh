#!/usr/bin/env bash
set -euo pipefail

TOOL_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_IMAGE=${BASE_IMAGE:-va-v4-realtime:torch241-cu121}
OUTPUT_IMAGE=${OUTPUT_IMAGE:-va-v4-realtime:torch241-cu121-opt2-compile}

docker image inspect "$BASE_IMAGE" >/dev/null
docker build \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --file "$TOOL_DIR/Dockerfile.torch-compile" \
  --tag "$OUTPUT_IMAGE" \
  "$TOOL_DIR"

docker run --rm --gpus all "$OUTPUT_IMAGE" bash -lc '
set -euo pipefail
command -v cc
command -v c++
cc --version | head -n 1
python - <<"PY"
import torch

class Smoke(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x) + 0.5 * x

model = torch.compile(Smoke().cuda().eval(), mode="reduce-overhead")
x = torch.ones(4096, device="cuda")
for _ in range(2):
    y = model(x)
torch.cuda.synchronize()
assert torch.isfinite(y).all()
print("TORCH_COMPILE_CUDA_SMOKE_OK")
PY
'

echo "V4_STAGE1_OPT2_COMPILE_IMAGE_OK"
echo "IMAGE=$OUTPUT_IMAGE"

