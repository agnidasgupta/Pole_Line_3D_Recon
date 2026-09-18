#!/usr/bin/env bash
# Host-side packaging only. It intentionally excludes NPZ and all model/checkpoint files.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/v4_full_ops_common.sh"
PTR="$V4_PROD_ROOT_HOST/LATEST_FULL_DATASET_RUN.txt"
ROOT=$(cat "$PTR" 2>/dev/null || true)
[[ -n "$ROOT" && -d "$ROOT" ]] || { echo "ERROR: no full-dataset V4 run found." >&2; exit 2; }
PACKROOT="$V4_PROD_ROOT_HOST/full_dataset_packages"; mkdir -p "$PACKROOT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
STAGE=$(mktemp -d "$PACKROOT/.all_data_package.XXXXXX")
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/manifests/stage1" "$STAGE/context"
cp -f "$ROOT/run_context.env" "$ROOT/all_sessions.tsv" "$ROOT/raw_slice_inventory.tsv" "$ROOT/phase_wall_times.tsv" "$STAGE/context/" 2>/dev/null || true
for f in PHASE1_STAGE1_OK.txt PHASE2_STAGE2_OK.txt PHASE3_STAGE3_OK.txt COMPLETED.txt FAILED.txt FULL_DATASET_SUMMARY.json FULL_DATASET_SUMMARY.txt; do
  [[ -f "$ROOT/$f" ]] && cp -f "$ROOT/$f" "$STAGE/"
done
[[ -d "$ROOT/logs" ]] && cp -a "$ROOT/logs" "$STAGE/"
[[ -d "$ROOT/metrics" ]] && cp -a "$ROOT/metrics" "$STAGE/"
[[ -d "$ROOT/timings" ]] && cp -a "$ROOT/timings" "$STAGE/"
[[ -d "$ROOT/stage1_inference" ]] && cp -a "$ROOT/stage1_inference" "$STAGE/"
[[ -d "$ROOT/stage2" ]] && cp -a "$ROOT/stage2" "$STAGE/"
[[ -d "$ROOT/stage3" ]] && cp -a "$ROOT/stage3" "$STAGE/"
# Preserve Stage1 manifests/completion records without copying Stage1 NPZ score caches.
if [[ -d "$ROOT/stage1" ]]; then
  while IFS= read -r f; do
    rel=${f#"$ROOT/stage1/"}; dest="$STAGE/manifests/stage1/$rel"; mkdir -p "$(dirname "$dest")"; cp -f "$f" "$dest"
  done < <(find "$ROOT/stage1" -type f \( -name 'stage1_manifest.csv' -o -name 'STAGE1_COMPLETED.json' \) -print)
fi
# Hard safety gate: never package model/checkpoint/cache files.
BAD=$(find "$STAGE" -type f \( -iname '*.npz' -o -iname '*.pt' -o -iname '*.pth' -o -iname '*.ckpt' -o -iname '*.joblib' -o -iname '*.onnx' -o -iname '*.engine' -o -iname '*.safetensors' \) -print -quit)
[[ -z "$BAD" ]] || { echo "ERROR: forbidden file entered package staging: $BAD" >&2; exit 3; }
cat > "$STAGE/PACKAGE_INFO.txt" <<INFO
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
source_run=$ROOT
deployment_fingerprint=$V4_DEPLOY_FINGERPRINT
runtime_mode=$V4_RUNTIME_MODE
batch_size=$V4_BATCH_SIZE
contains_npz=0
contains_model_files=0
includes_stage1_inference_csv_gz=1
includes_stage2_reconstruction_csv=1
includes_stage3_reconstruction_csv_json_logs=1
includes_metrics_and_timings=1
INFO
ARCH="$PACKROOT/v4_all_data_results_${V4_DEPLOY_SHORT}_${STAMP}.tar.gz"
tar -C "$STAGE" -czf "$ARCH" .
sha256sum "$ARCH" > "$ARCH.sha256"
printf '%s\n' "$ARCH" > "$PACKROOT/LATEST_ALL_DATA_ARCHIVE.txt"
printf '%s\n' "$ARCH.sha256" > "$PACKROOT/LATEST_ALL_DATA_SHA256.txt"
echo "Results archive: $ARCH"
echo "SHA256 file:    $ARCH.sha256"
echo "Archive size:   $(du -h "$ARCH" | awk '{print $1}')"
echo "NPZ/model safety check: PASS"
