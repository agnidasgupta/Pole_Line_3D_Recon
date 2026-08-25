#!/usr/bin/env bash
# Host-shell-only test for Nebius path/session discovery. It invokes no Python.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
IN="$TMP/voxel_csv_combined"
OUT="$TMP/outputs"
mkdir -p "$IN/GeoA/session1_slice1" "$IN/GeoA/session1_slice2" \
         "$IN/GeoB/session2_slice1" "$IN/GeoB/session2_slice3" "$IN/GeoB/session2_slice5" \
         "$IN/GeoC/session3_slice1" "$IN/GeoC/session3_slice2" "$IN/GeoC/session3_slice3" "$IN/GeoC/session3_slice4"
for p in \
  "$IN/GeoA/session1_slice1/a.csv" "$IN/GeoA/session1_slice2/a.csv" \
  "$IN/GeoB/session2_slice1/a.csv" "$IN/GeoB/session2_slice3/a.csv" "$IN/GeoB/session2_slice5/a.csv" \
  "$IN/GeoC/session3_slice1/a.csv" "$IN/GeoC/session3_slice2/a.csv" "$IN/GeoC/session3_slice3/a.csv" "$IN/GeoC/session3_slice4/a.csv" \
  "$IN/GeoC/session3_slice4/duplicate.csv"; do printf 'x,y,z\n0,0,0\n' > "$p"; done
mkdir -p "$OUT/poleline_voxel_run_session_groups/precision_v4/train" \
         "$OUT/poleline_voxel_run_session_groups/precision_v4/full_val" \
         "$OUT/poleline_voxel_run_session_groups/v4_realtime/stage2_refiner"
printf 'model\n' > "$OUT/poleline_voxel_run_session_groups/precision_v4/train/precision_best.pt"
printf '{}\n' > "$OUT/poleline_voxel_run_session_groups/precision_v4/full_val/calibration.json"
printf 'bundle\n' > "$OUT/poleline_voxel_run_session_groups/v4_realtime/stage2_refiner/local_refiner_bundle.joblib"
(
  export HOST_INPUT_DIR="$IN" HOST_OUTPUTS="$OUT"
  unset SESSION_FILTER MODEL_HOST CAL_HOST STAGE2_BUNDLE_HOST V4_PROD_ROOT_HOST V4_SESSION_ROOT_HOST V4_RUN_ROOT_HOST || true
  # shellcheck source=/dev/null
  source "$HERE/v4_nebius_common.sh"
  [[ "$V4_SESSION_FILTER" == 'GeoB/session2' ]] || { echo "unexpected selected session: $V4_SESSION_FILTER" >&2; exit 1; }
  [[ -s "$V4_SESSION_INVENTORY_HOST" ]] || { echo 'session inventory not persisted' >&2; exit 1; }
  [[ -s "$V4_RUN_ROOT_HOST/run_context.env" ]] || { echo 'run context not persisted' >&2; exit 1; }
  [[ "$(v4_to_container_output "$V4_RUN_ROOT_HOST")" == /outputs/* ]] || { echo 'container output mapping failed' >&2; exit 1; }
  [[ "$V4_DEPLOY_FINGERPRINT" =~ ^[0-9a-f]{64}$ ]] || { echo 'deployment fingerprint invalid' >&2; exit 1; }
)
echo 'V4_NEBIUS_DISCOVERY_SMOKE_OK selected_largest_valid_session=1 duplicate_session_rejected=1 persistent_context=1'
