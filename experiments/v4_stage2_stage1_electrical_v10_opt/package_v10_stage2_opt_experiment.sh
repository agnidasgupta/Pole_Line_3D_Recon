#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT=$(head -n 1 /home/agni/LATEST_V10_STAGE2_OPT_RUN.txt)
[ -d "$RUN_ROOT" ] || { echo "ERROR: run root missing: $RUN_ROOT" >&2; exit 1; }
[ -s "$RUN_ROOT/STAGE2_ONLY_COMPLETE.txt" ] || { echo "ERROR: Stage2 completion marker missing" >&2; exit 1; }
accepted=$(find "$RUN_ROOT/status" -maxdepth 1 -type f -name '*.stage2.ok' | wc -l | tr -d ' ')
failed=$(find "$RUN_ROOT/status" -maxdepth 1 -type f -name '*.stage2.failed' | wc -l | tr -d ' ')
validated=$(find "$RUN_ROOT/status" -maxdepth 1 -type f -name '*.voxel_support_validation.json' | wc -l | tr -d ' ')
equivalent=$(find "$RUN_ROOT/status" -maxdepth 1 -type f -name '*.quality_equivalence.json' | wc -l | tr -d ' ')
[ -s "$RUN_ROOT/STAGE2_TIMING_SESSION_AVERAGES.txt" ] || { echo "ERROR: timing summary missing" >&2; exit 1; }
[ "$accepted" -eq 30 ] && [ "$failed" -eq 0 ] && [ "$validated" -eq 30 ] && [ "$equivalent" -eq 30 ] || {
  echo "ERROR: not ready: accepted=$accepted failed=$failed validated=$validated equivalent=$equivalent" >&2; exit 1;
}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
archive=/home/agni/v10_stage2_opt_experiment_results_${stamp}.tar.gz
tar -czf "$archive" -C "$(dirname "$RUN_ROOT")" "$(basename "$RUN_ROOT")"
(cd /home/agni && sha256sum "$(basename "$archive")" > "$(basename "$archive").sha256")
printf '%s\n' "$archive" > /home/agni/LATEST_V10_STAGE2_OPT_ARCHIVE.txt
echo "PACKAGE_OK"
echo "ARCHIVE=$archive"
echo "SIZE=$(du -h "$archive" | awk '{print $1}')"
cat "$archive.sha256"
