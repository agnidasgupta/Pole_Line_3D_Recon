#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT=$(head -n 1 /home/agni/LATEST_V4_STAGE1_OPT2_RUN.txt)
[ -d "$RUN_ROOT" ] || { echo "ERROR: run root missing: $RUN_ROOT" >&2; exit 1; }
[ -s "$RUN_ROOT/STAGE1_OPT2_PRODUCTION_EQUIVALENT_COMPLETE.txt" ] || { echo "ERROR: completion marker missing" >&2; exit 1; }
[ -s "$RUN_ROOT/STAGE1_TIMING_SESSION_AVERAGES.txt" ] || { echo "ERROR: timing summary missing" >&2; exit 1; }
accepted=$(find "$RUN_ROOT/status" -maxdepth 1 -type f -name '*.stage1.ok' | wc -l | tr -d ' ')
failed=$(find "$RUN_ROOT/status" -maxdepth 1 -type f -name '*.stage1.failed' | wc -l | tr -d ' ')
guarded=$(find "$RUN_ROOT/status" -maxdepth 1 -type f -name '*.production_equivalence.json' | wc -l | tr -d ' ')
[ "$accepted" -eq 30 ] && [ "$failed" -eq 0 ] && [ "$guarded" -eq 30 ] || {
  echo "ERROR: not ready: accepted=$accepted failed=$failed production_equivalent=$guarded" >&2
  exit 1
}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
archive=/home/agni/v4_stage1_opt2_production_equivalent_results_${stamp}.tar.gz
HOST_OUTPUTS=/workspace/voxel_poleline/outputs
case "$RUN_ROOT" in
  "$HOST_OUTPUTS"/*) relative=${RUN_ROOT#"$HOST_OUTPUTS"/} ;;
  *) echo "ERROR: run root is outside $HOST_OUTPUTS: $RUN_ROOT" >&2; exit 1 ;;
esac
tar -czf "$archive" -C "$HOST_OUTPUTS" "$relative"
(cd /home/agni && sha256sum "$(basename "$archive")" > "$(basename "$archive").sha256")
printf '%s\n' "$archive" > /home/agni/LATEST_V4_STAGE1_OPT2_ARCHIVE.txt
echo "PACKAGE_OK"
echo "ARCHIVE=$archive"
echo "SIZE=$(du -h "$archive" | awk '{print $1}')"
cat "$archive.sha256"
