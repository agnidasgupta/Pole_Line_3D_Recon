#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?Set REPO to the experimental Git worktree}"
: "${UNITY_PROJECT:?Set UNITY_PROJECT to the successfully built Unity project}"

OPS="$REPO/ops/v4_stage12_unity_native"
BUILD_DIR="${UNITY_BUILD_DIR:-$UNITY_PROJECT/Builds/Linux}"
PLAYER="$BUILD_DIR/V10Stage12GpuPlayer"
PLAYER_DATA="$BUILD_DIR/V10Stage12GpuPlayer_Data"
MODEL_DIR="$UNITY_PROJECT/Assets/V10Stage12/Models"
BUILD_LOG="$BUILD_DIR/V10Stage12Build.log"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/Downloads}"

test -x "$PLAYER" || { echo "ERROR: successful GPU player missing: $PLAYER" >&2; exit 1; }
test -d "$PLAYER_DATA" || { echo "ERROR: Unity player data missing: $PLAYER_DATA" >&2; exit 1; }
test -s "$BUILD_LOG" || { echo "ERROR: Unity build log missing: $BUILD_LOG" >&2; exit 1; }
test -d "$MODEL_DIR" || { echo "ERROR: model assets missing: $MODEL_DIR" >&2; exit 1; }
for required in \
  v10_stage1_voxelnet3d_fp32.onnx \
  v10_stage12_sidecar.json \
  v10_stage2_refiner_trees.bytes
do
  test -s "$MODEL_DIR/$required" || { echo "ERROR: missing model asset: $MODEL_DIR/$required" >&2; exit 1; }
done

bash "$OPS/scripts/verify_v10_unity_native_source.sh"
git -C "$REPO" diff --check
branch=$(git -C "$REPO" branch --show-current)
[[ "$branch" == "v4-stage12-unity-native-v10" ]] || {
  echo "ERROR: expected experimental branch v4-stage12-unity-native-v10; found $branch" >&2
  exit 1
}
if [[ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]]; then
  echo "ERROR: tracked Git files are not committed; commit the exact handoff source first" >&2
  git -C "$REPO" status --short
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
git_sha=$(git -C "$REPO" rev-parse HEAD)
short_sha=$(printf '%s' "$git_sha" | cut -c1-12)
name="v10_unity_native_stage12_realtime_${stamp}_${short_sha}"
temporary=$(mktemp -d "${TMPDIR:-/tmp}/v10_unity_handoff.XXXXXX")
root="$temporary/$name"

cleanup() {
  if [[ -n "${temporary:-}" && -d "$temporary" ]]; then
    rm -rf -- "$temporary"
  fi
}
trap cleanup EXIT

mkdir -p "$root/runtime/Linux" "$root/model_assets/Assets/V10Stage12/Models" \
  "$root/scripts" "$root/source/ops" "$root/provenance"

rsync -a --exclude '*_BurstDebugInformation_DoNotShip*' "$BUILD_DIR/" "$root/runtime/Linux/"
rsync -a "$MODEL_DIR/" "$root/model_assets/Assets/V10Stage12/Models/"
rsync -a "$OPS/" "$root/source/ops/v4_stage12_unity_native/"
rsync -a "$OPS/scripts/launch_v10_unity_realtime_handoff.sh" "$root/scripts/"
rsync -a "$OPS/scripts/monitor_v10_unity_realtime_handoff.sh" "$root/scripts/"
rsync -a "$OPS/REALTIME_HANDOFF_README.md" "$root/README.md"
rsync -a "$BUILD_LOG" "$root/provenance/V10Stage12Build.log"

printf '%s\n' "$git_sha" > "$root/provenance/GIT_COMMIT.txt"
printf '%s\n' "$branch" > "$root/provenance/GIT_BRANCH.txt"
git -C "$REPO" status --short > "$root/provenance/GIT_STATUS.txt"
printf '%s\n' '6000.3.15f1' > "$root/provenance/UNITY_EDITOR_VERSION.txt"
printf '%s\n' 'com.unity.ai.inference=2.6.1' > "$root/provenance/SENTIS_VERSION.txt"
printf '%s\n' 'backend=GPUCompute' 'precision=FP32' 'graphics=Vulkan' 'batch_size=12' \
  > "$root/provenance/RUNTIME_CONFIGURATION.txt"

test -x "$root/runtime/Linux/V10Stage12GpuPlayer"
test -d "$root/runtime/Linux/V10Stage12GpuPlayer_Data"
find "$root" -type f ! -name MANIFEST.txt -print | sed "s#^$root/##" | LC_ALL=C sort \
  > "$root/MANIFEST.txt"

(
  cd "$root"
  find . -type f ! -name CHECKSUMS.sha256 -print | LC_ALL=C sort | while IFS= read -r file
  do
    shasum -a 256 "$file"
  done > CHECKSUMS.sha256
)

archive="$OUTPUT_DIR/$name.tar.gz"
tar -C "$temporary" -czf "$archive" "$name"
shasum -a 256 "$archive" > "$archive.sha256"
printf '%s\n' "$archive" > "$OUTPUT_DIR/LATEST_V10_UNITY_REALTIME_HANDOFF.txt"

echo "HANDOFF_PACKAGE_OK=$archive"
echo "HANDOFF_SHA256=$archive.sha256"
echo "GIT_COMMIT=$git_sha"
du -sh "$archive"
cat "$archive.sha256"
