#!/usr/bin/env bash
set -euo pipefail

: "${EXPORT_ROOT:?Set EXPORT_ROOT to the completed V10 Unity export run directory}"
: "${REPO:?Set REPO to the experimental repository worktree}"

CODE="$REPO/ops/v4_stage12_unity_native"
FP32=$(find "$EXPORT_ROOT" -type f -name 'v10_stage1_voxelnet3d_fp32.onnx' -print -quit)
FP16=$(find "$EXPORT_ROOT" -type f -name 'v10_stage1_voxelnet3d_fp16.onnx' -print -quit || true)
SIDECAR=$(find "$EXPORT_ROOT" -type f -name 'v10_stage12_sidecar.json' -print -quit)
TREES=$(find "$EXPORT_ROOT" -type f -name 'v10_stage2_refiner_trees.json' -print -quit)
test -s "$FP32" && test -s "$SIDECAR" && test -s "$TREES"
command -v jq >/dev/null

check_sha() {
  local file=$1
  local expected=$2
  local actual
  actual=$(sha256sum "$file" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || {
    echo "SHA256 mismatch for $file" >&2
    exit 1
  }
}

check_sha "$FP32" "$(jq -r '.onnx.models.fp32.sha256' "$SIDECAR")"
check_sha "$TREES" "$(jq -r '.stage2.refiner_tree_metadata.sha256' "$SIDECAR")"
if [[ -n "$FP16" ]]; then
  check_sha "$FP16" "$(jq -r '.onnx.models.fp16.sha256' "$SIDECAR")"
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
WORK=$(mktemp -d /home/agni/v10_unity_assets.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/Assets/V10Stage12/Models"
cp "$FP32" "$WORK/Assets/V10Stage12/Models/"
[[ -n "$FP16" ]] && cp "$FP16" "$WORK/Assets/V10Stage12/Models/"
cp "$SIDECAR" "$WORK/Assets/V10Stage12/Models/"

docker run --rm \
  --mount "type=bind,source=$CODE,target=/code,readonly" \
  --mount "type=bind,source=$TREES,target=/input/trees.json,readonly" \
  --mount "type=bind,source=$WORK/Assets/V10Stage12/Models,target=/output" \
  va-v4-realtime:torch241-cu121 \
  python /code/tools/export_v10_stage2_tree_binary.py \
    --input-json /input/trees.json \
    --output-bytes /output/v10_stage2_refiner_trees.bytes

archive="/home/agni/v10_unity_native_model_assets_${STAMP}.tar.gz"
tar -C "$WORK" -czf "$archive" Assets
sha256sum "$archive" > "$archive.sha256"
printf '%s\n' "$archive" > /home/agni/LATEST_V10_UNITY_NATIVE_MODEL_ASSETS.txt
echo "ASSET_PACKAGE_OK=$archive"
cat "$archive.sha256"
