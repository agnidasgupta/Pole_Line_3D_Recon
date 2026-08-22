#!/usr/bin/env bash
set -euo pipefail
WORK_DIR="${WORK_DIR:-/workspace/voxel_poleline/outputs/poleline_voxel_run_session_groups}"
RUN="${RUN:-$WORK_DIR/v62_teacher_recall}"
echo "RUN=$RUN"
for f in TRAINING_COMPLETED.json inference_all/INFERENCE_COMPLETED.json stage3_reconstruction/COMPLETED.json COMPLETED.json; do
  if [[ -f "$RUN/$f" ]]; then echo "OK      $f"; else echo "PENDING $f"; fi
done
for d in stage1_train stage1_test stage2_component_mining stage2_refiner test_inference inference_all stage3_reconstruction diagnostics training_diagnostics; do
  [[ -d "$RUN/$d" ]] && printf '%-28s ' "$d" && du -sh "$RUN/$d" | awk '{print $1}'
done
