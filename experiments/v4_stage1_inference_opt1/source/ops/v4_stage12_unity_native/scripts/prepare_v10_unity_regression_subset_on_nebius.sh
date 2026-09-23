#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?Set REPO to the experimental repository worktree}"
: "${STAGE1_ROOT:?Set STAGE1_ROOT to the completed all-session Stage1 root}"
: "${ACCEPTED_ROOT:?Set ACCEPTED_ROOT to the accepted opt1-fix2 run root}"

OUTPUTS=/workspace/voxel_poleline/outputs
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SUBSET="$OUTPUTS/poleline_voxel_run_session_groups/v4_stage23_quality/unity_native_v10_regression_inputs_${STAMP}"

docker run --rm \
  --mount "type=bind,source=$REPO/ops/v4_stage12_unity_native,target=/code,readonly" \
  --mount "type=bind,source=$OUTPUTS,target=/outputs" \
  va-v4-realtime:torch241-cu121 \
  python /code/tools/prepare_unity_regression_subset.py \
    --stage1-root "${STAGE1_ROOT/#$OUTPUTS//outputs}" \
    --accepted-root "${ACCEPTED_ROOT/#$OUTPUTS//outputs}" \
    --output-root "${SUBSET/#$OUTPUTS//outputs}"

printf '%s\n' "$SUBSET/input_manifest.csv" > /home/agni/LATEST_V10_UNITY_REGRESSION_MANIFEST.txt
printf '%s\n' "$SUBSET/reference" > /home/agni/LATEST_V10_UNITY_REGRESSION_REFERENCE.txt
echo "REGRESSION_MANIFEST=$SUBSET/input_manifest.csv"
echo "REGRESSION_REFERENCE=$SUBSET/reference"
