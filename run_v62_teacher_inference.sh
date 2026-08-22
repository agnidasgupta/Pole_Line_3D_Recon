#!/usr/bin/env bash
set -euo pipefail
WORK_DIR="${WORK_DIR:-/outputs/poleline_voxel_run_session_groups}"
RAW_INPUT_DIR="${RAW_INPUT_DIR:-/data/voxel_csv_combined}"
V62_DIR="${V62_DIR:-$WORK_DIR/v62_teacher_recall}"
DATASET="$V62_DIR/dataset_local"
SELECTED="$V62_DIR/selected"
REFINER="$V62_DIR/stage2_refiner"
ALLINF="${OUTPUT_DIR:-$V62_DIR/inference_all}"

mkdir -p "$V62_DIR" "$ALLINF"
test -d "$RAW_INPUT_DIR" || { echo "Missing raw input: $RAW_INPUT_DIR"; exit 2; }
test -f "$SELECTED/v6_stage1_selected.pt" || { echo "Missing selected model"; exit 2; }
test -f "$SELECTED/calibration.json" || { echo "Missing calibration"; exit 2; }
test -f "$REFINER/local_refiner_bundle.joblib" || { echo "Missing Stage-2 refiner"; exit 2; }
rm -f "$ALLINF/INFERENCE_FAILED.json" "$ALLINF/INFERENCE_COMPLETED.json"

if [[ -f "$DATASET/manifests/summary.json" ]]; then
  GRID=$(python - "$DATASET/manifests/summary.json" <<'PY'
import json,sys
print(*json.load(open(sys.argv[1]))['grid_size_xyz'])
PY
)
else
  GRID="${GRID_SIZE:-400 400 200}"
fi

export INPUT_DIR="$RAW_INPUT_DIR"
export OUTPUT_DIR="$ALLINF"
export MODEL_PATH="$SELECTED/v6_stage1_selected.pt"
export CALIBRATION_JSON="$SELECTED/calibration.json"
export LOCAL_REFINER_BUNDLE="$REFINER/local_refiner_bundle.joblib"
export GRID_SIZE="$GRID"
export POLE_CANDIDATE_THRESHOLD="${POLE_CANDIDATE_THRESHOLD:-0.15}"
export LINE_CANDIDATE_THRESHOLD="${LINE_CANDIDATE_THRESHOLD:-0.08}"
export LINE_WEAK_THRESHOLD="${LINE_WEAK_THRESHOLD:-0.04}"
export LINE_COMPETITION_RATIO="${LINE_COMPETITION_RATIO:-0.55}"
export LINE_MIN_VOXELS="${LINE_MIN_VOXELS:-3}"
export RESUME="${RESUME:-1}"

set +e
./run_v62_inference.sh
code=$?
set -e
if [[ "$code" -ne 0 ]]; then
  python - "$ALLINF" "$code" <<'PY'
import json,os,sys,time
out,code=sys.argv[1:]
os.makedirs(out,exist_ok=True)
json.dump({'completed':False,'exit_code':int(code),'failed_at':time.strftime('%F %T')},open(os.path.join(out,'INFERENCE_FAILED.json'),'w'),indent=2)
PY
  exit "$code"
fi

python - "$ALLINF" <<'PY'
import json,os,sys,time
out=sys.argv[1]
os.makedirs(out,exist_ok=True)
json.dump({'completed':True,'completed_at':time.strftime('%F %T'),'output_dir':out},open(os.path.join(out,'INFERENCE_COMPLETED.json'),'w'),indent=2)
PY
rm -f "$ALLINF/INFERENCE_FAILED.json"
echo "V6.2 TEACHER-RECALL INFERENCE COMPLETE"
