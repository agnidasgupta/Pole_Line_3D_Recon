#!/usr/bin/env bash
set -euo pipefail
WORK_DIR="${WORK_DIR:-/outputs/poleline_voxel_run_session_groups}"
V62_DIR="${V62_DIR:-$WORK_DIR/v62_teacher_recall}"
DIAG="$V62_DIR/diagnostics"
mkdir -p "$V62_DIR" "$DIAG"

if [[ -f "$V62_DIR/TRAINING_COMPLETED.json" ]]; then
  echo "[all] training already complete; skipping"
else
  ./run_v62_teacher_training.sh
fi

if [[ -f "$V62_DIR/inference_all/INFERENCE_COMPLETED.json" && -f "$V62_DIR/inference_all/COMPLETED.json" ]]; then
  echo "[all] inference already complete; skipping"
else
  ./run_v62_teacher_inference.sh
fi

if [[ -f "$V62_DIR/stage3_reconstruction/COMPLETED.json" ]]; then
  echo "[all] reconstruction already complete; skipping"
else
  ./run_v62_teacher_reconstruction.sh
fi

python collect_v62_diagnostics.py --stage1_test "$V62_DIR/stage1_test/full_scene_metrics.json" --stage2_metrics "$V62_DIR/stage2_refiner/local_refiner_metrics.json" --inference_metrics "$V62_DIR/inference_all/inference_metrics.json" --stage3_completed "$V62_DIR/stage3_reconstruction/COMPLETED.json" --output_dir "$DIAG"
python - "$V62_DIR" <<'PY'
import json,os,sys,time
out=sys.argv[1]; os.makedirs(out,exist_ok=True); json.dump({'completed':True,'completed_at':time.strftime('%F %T'),'training':True,'inference':True,'reconstruction':True},open(os.path.join(out,'COMPLETED.json'),'w'),indent=2)
PY
echo "V6.2 TEACHER-RECALL ALL STAGES COMPLETE"
