#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/v4_full_ops_common.sh"
PTR="$V4_PROD_ROOT_HOST/LATEST_FULL_DATASET_RUN.txt"
ROOT=$(cat "$PTR" 2>/dev/null || true)
[[ -n "$ROOT" && -d "$ROOT" ]] || { echo "No full-dataset V4 run found."; exit 1; }
echo "Run root: $ROOT"
[[ -f "$ROOT/run_context.env" ]] && cat "$ROOT/run_context.env"
echo "--- markers ---"
for f in PHASE1_STAGE1_OK.txt PHASE2_STAGE2_OK.txt PHASE3_STAGE3_OK.txt COMPLETED.txt FAILED.txt; do
  [[ -f "$ROOT/$f" ]] && { echo "### $f"; cat "$ROOT/$f"; }
done
NAME=$(cat "$ROOT/current_container.txt" 2>/dev/null || true)
if [[ -n "$NAME" ]]; then
  echo "--- container ---"
  "${V4_DOCKER[@]}" ps -a --filter "name=^/${NAME}$" --format 'table {{.Names}}\t{{.Status}}\t{{.ID}}' || true
fi
echo "--- session/slice counts ---"
[[ -f "$ROOT/all_sessions.tsv" ]] && awk -F '\t' 'NR>1{n++;s+=$2}END{printf "sessions=%d slices=%d\n",n,s}' "$ROOT/all_sessions.tsv"
echo "--- recent log ---"
LOG=$(find "$ROOT" -maxdepth 1 -type f -name 'all_data_*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)
[[ -n "$LOG" ]] && tail -80 "$LOG" || true
echo "--- summary ---"
[[ -f "$ROOT/FULL_DATASET_SUMMARY.txt" ]] && cat "$ROOT/FULL_DATASET_SUMMARY.txt" || true
