#!/usr/bin/env bash
set -euo pipefail

# Package only compact E5b diagnostics and timing results. The raw Stage 1 NPZ
# artifacts, per-voxel inference CSV.GZ files, model/checkpoint files, datasets,
# and binary Nsight reports are intentionally excluded.

RUN_POINTER=${RUN_POINTER:-/home/agni/V4_STAGE1_OPT2_E5B_FULL30_ROOT.txt}
PROFILE_POINTER=${PROFILE_POINTER:-/home/agni/LATEST_V4_STAGE1_OPT2_PROFILE.txt}
MAX_ARCHIVE_BYTES=${MAX_ARCHIVE_BYTES:-104857600}

fail() { echo "ERROR: $*" >&2; exit 1; }
[ -s "$RUN_POINTER" ] || fail "run pointer missing: $RUN_POINTER"
[ -s "$PROFILE_POINTER" ] || fail "profile pointer missing: $PROFILE_POINTER"
RUN_ROOT=${RUN_ROOT:-$(head -n 1 "$RUN_POINTER")}
PROFILE_ROOT=${PROFILE_ROOT:-$(head -n 1 "$PROFILE_POINTER")}
[ -d "$RUN_ROOT" ] || fail "run root missing: $RUN_ROOT"
[ -d "$PROFILE_ROOT" ] || fail "profile root missing: $PROFILE_ROOT"

for path in \
  "$RUN_ROOT/RUN_INFO.txt" \
  "$RUN_ROOT/STAGE1_OPT2_PRODUCTION_EQUIVALENT_COMPLETE.txt" \
  "$RUN_ROOT/STAGE1_TIMING_SESSION_AVERAGES.txt" \
  "$PROFILE_ROOT/PROFILE_SUMMARY.json" \
  "$PROFILE_ROOT/NSYS_STATS.txt" \
  "$PROFILE_ROOT/stage1_opt2_nsys.nsys-rep"; do
  [ -s "$path" ] || fail "required result missing: $path"
done

grep -qx 'cache_coordinate_channels=0' "$RUN_ROOT/RUN_INFO.txt" || \
  fail "full run unexpectedly enabled rejected E5"
grep -qx 'cache_reference_coordinate_channels=1' "$RUN_ROOT/RUN_INFO.txt" || \
  fail "full run did not enable E5b"
grep -Eq '"cache_reference_coordinate_channels"[[:space:]]*:[[:space:]]*true' \
  "$PROFILE_ROOT/PROFILE_SUMMARY.json" || fail "profile is not an E5b capture"

accepted=$(find "$RUN_ROOT/status" -maxdepth 1 -type f -name '*.stage1.ok' | wc -l | tr -d ' ')
failed=$(find "$RUN_ROOT/status" -maxdepth 1 -type f -name '*.stage1.failed' | wc -l | tr -d ' ')
guarded=$(find "$RUN_ROOT/status" -maxdepth 1 -type f -name '*.production_equivalence.json' | wc -l | tr -d ' ')
[ "$accepted" -eq 30 ] && [ "$failed" -eq 0 ] && [ "$guarded" -eq 30 ] || \
  fail "not ready: accepted=$accepted failed=$failed production_equivalent=$guarded"

slices=$(find "$RUN_ROOT/timings/stage1" -maxdepth 1 -type f -name '*.csv' -print0 | \
  xargs -0 -r awk 'FNR > 1 {count++} END {print count+0}')
[ "$slices" -eq 3738 ] || fail "expected 3738 timing rows, found $slices"

stamp=$(date -u +%Y%m%dT%H%M%SZ)
archive=/home/agni/v4_stage1_opt2_e5b_slim_results_${stamp}.tar.gz
tmp=$(mktemp -d /tmp/v4_stage1_opt2_e5b_slim.XXXXXX)
trap 'rm -rf -- "$tmp"' EXIT
bundle="$tmp/v4_stage1_opt2_e5b_slim_results_${stamp}"
run_dest="$bundle/full30"
profile_dest="$bundle/nsight_text"
mkdir -p "$run_dest" "$profile_dest"

copy_relative() {
  local root=$1
  local source=$2
  local destination=$3
  local relative=${source#"$root"/}
  [ "$relative" != "$source" ] || fail "source is outside expected root: $source"
  mkdir -p "$destination/$(dirname "$relative")"
  cp -p -- "$source" "$destination/$relative"
}

for name in \
  RUN_INFO.txt session_map.tsv STAGE1_TIMING_SESSION_AVERAGES.txt \
  PHASE1_STAGE1_OK.txt STAGE1_OPT2_PRODUCTION_EQUIVALENT_COMPLETE.txt \
  FILE_INVENTORY.txt; do
  [ -f "$RUN_ROOT/$name" ] && cp -p -- "$RUN_ROOT/$name" "$run_dest/$name"
done

while IFS= read -r -d '' file; do
  copy_relative "$RUN_ROOT" "$file" "$run_dest"
done < <(
  find "$RUN_ROOT/timings/stage1" -maxdepth 1 -type f -name '*.csv' -print0
  find "$RUN_ROOT/status" -maxdepth 1 -type f \
    \( -name '*.stage1.ok' -o -name '*.progress.json' -o -name '*.production_equivalence.json' \) \
    -print0
  find "$RUN_ROOT/stage1" -type f -name 'stage1_manifest.csv' -print0
  find "$RUN_ROOT/stage1_inference" -type f \
    \( -name 'inference_manifest.csv' -o -name 'stage1_metrics_by_slice.csv' -o -name 'stage1_metrics_summary.json' \) \
    -print0
  if [ -d "$RUN_ROOT/metrics" ]; then
    find "$RUN_ROOT/metrics" -type f -size -10M -print0
  fi
)

for name in \
  PROFILE_SUMMARY.json NSYS_STATS.txt NSYS_PROFILE.log \
  MODEL_INVENTORY.txt MODEL_INVENTORY.json PROFILER_TOOLS.txt SHA256SUMS.txt; do
  [ -f "$PROFILE_ROOT/$name" ] && cp -p -- "$PROFILE_ROOT/$name" "$profile_dest/$name"
done

{
  echo "V4 Stage1 Opt2 E5b compact result package"
  echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "full_run_root=$RUN_ROOT"
  echo "profile_root=$PROFILE_ROOT"
  echo "accepted_sessions=$accepted"
  echo "production_equivalent_sessions=$guarded"
  echo "timed_slices=$slices"
  echo "excluded=NPZ,PT,PTH,CKPT,ONNX,ENGINE,SAFETENSORS,CSV.GZ,raw datasets,inference_rows,NSYS-REP,SQLite"
} > "$bundle/PACKAGE_INFO.txt"

find "$bundle" -type f -printf '%P\t%s\n' | sort > "$bundle/PACKAGE_FILE_INVENTORY.txt"

if find "$bundle" -type f | grep -Eiq '\.(npz|pt|pth|ckpt|onnx|engine|safetensors|csv\.gz|nsys-rep|sqlite)$|/inference_rows/'; then
  find "$bundle" -type f | grep -Ei '\.(npz|pt|pth|ckpt|onnx|engine|safetensors|csv\.gz|nsys-rep|sqlite)$|/inference_rows/' >&2
  fail "forbidden large or binary artifacts entered the slim bundle"
fi

tar -C "$tmp" -czf "$archive" "$(basename "$bundle")"
listing="$tmp/archive_listing.txt"
tar -tzf "$archive" > "$listing" || fail "created archive cannot be read"
if grep -Eiq '\.(npz|pt|pth|ckpt|onnx|engine|safetensors|csv\.gz|nsys-rep|sqlite)$|/inference_rows/' "$listing"; then
  grep -Ei '\.(npz|pt|pth|ckpt|onnx|engine|safetensors|csv\.gz|nsys-rep|sqlite)$|/inference_rows/' "$listing" >&2
  fail "forbidden artifacts found in final archive"
fi

archive_bytes=$(stat -c '%s' "$archive")
[ "$archive_bytes" -le "$MAX_ARCHIVE_BYTES" ] || \
  fail "archive is unexpectedly large: $archive_bytes bytes > $MAX_ARCHIVE_BYTES"
(cd /home/agni && sha256sum "$(basename "$archive")" > "$(basename "$archive").sha256")
printf '%s\n' "$archive" > /home/agni/LATEST_V4_STAGE1_OPT2_E5B_SLIM_ARCHIVE.txt
printf '%s\n' "$archive" > /home/agni/LATEST_V4_STAGE1_OPT2_ARCHIVE.txt

echo "PACKAGE_OK"
echo "ARCHIVE=$archive"
echo "SIZE_BYTES=$archive_bytes"
echo "ACCEPTED=$accepted FAILED=$failed PRODUCTION_EQUIVALENT=$guarded SLICES=$slices"
cat "$archive.sha256"
