#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 4 ]]; then echo 'usage: v4_preflight_inside_docker.sh RUN_ROOT MODEL CAL STAGE2' >&2; exit 2; fi
RUN_ROOT=$1; MODEL=$2; CAL=$3; STAGE2=$4
mkdir -p "$RUN_ROOT"
FAIL="$RUN_ROOT/PREFLIGHT_FAILED.txt"; OK="$RUN_ROOT/PREFLIGHT_OK.txt"
trap 'rc=$?; printf "failed_utc=%s\nexit_code=%s\nlast_command=%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$BASH_COMMAND" > "$FAIL"; exit "$rc"' ERR
rm -f "$FAIL" "$OK"
test -s "$MODEL"; test -s "$CAL"; test -s "$STAGE2"
bash v4_code_validation_inside_docker.sh
python - <<'PY'
import torch, scipy, sklearn, pandas, numpy
assert torch.cuda.is_available(), 'CUDA is not available inside Docker'
print('GPU=',torch.cuda.get_device_name(0))
print('torch=',torch.__version__,'cuda=',torch.version.cuda,'bf16=',torch.cuda.is_bf16_supported())
print('scipy=',scipy.__version__,'sklearn=',sklearn.__version__,'pandas=',pandas.__version__,'numpy=',numpy.__version__)
PY
printf 'V4_REALTIME_NEBIUS_PREFLIGHT_OK\ncompleted_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OK"
cat "$OK"
