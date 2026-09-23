#!/usr/bin/env bash
# Common non-interactive Nebius runtime discovery for V4 production.
# Host-side shell only: every Python command is executed inside Docker.
set -euo pipefail

V4_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
V4_REPO_ROOT=$(cd "$V4_HERE/.." && pwd)
V4_IMAGE=${IMAGE:-va-v4-realtime:torch241-cu121}
if docker info >/dev/null 2>&1; then V4_DOCKER=(docker); else V4_DOCKER=(sudo docker); fi

v4_abs_dir() { (cd "$1" && pwd); }

v4_first_existing_dir() {
  local p
  for p in "$@"; do
    [[ -n "$p" && -d "$p" ]] && { v4_abs_dir "$p"; return 0; }
  done
  return 1
}

v4_first_existing_file() {
  local preferred=$1 pattern=$2 root=$3 found
  if [[ -n "$preferred" && -s "$preferred" ]]; then printf '%s\n' "$preferred"; return 0; fi
  found=$(find "$root" -type f -name "$pattern" -size +0c -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)
  [[ -n "$found" ]] || return 1
  printf '%s\n' "$found"
}

v4_choose_outputs_root() {
  # Explicit override is supported, but never required. Otherwise prefer the root
  # that already contains the V4 model/calibration/refiner artifacts.
  if [[ -n ${HOST_OUTPUTS:-} ]]; then
    [[ -d "$HOST_OUTPUTS" ]] || { echo "ERROR: HOST_OUTPUTS does not exist: $HOST_OUTPUTS" >&2; return 2; }
    v4_abs_dir "$HOST_OUTPUTS"; return 0
  fi
  local p score best='' best_score=-1
  for p in "$V4_REPO_ROOT/outputs" /workspace/voxel_poleline/outputs /outputs /data/outputs; do
    [[ -d "$p" ]] || continue
    score=0
    find "$p" -type f -name precision_best.pt -size +0c -print -quit 2>/dev/null | grep -q . && score=$((score+4)) || true
    find "$p" -type f -name calibration.json -size +0c -print -quit 2>/dev/null | grep -q . && score=$((score+2)) || true
    find "$p" -type f -name local_refiner_bundle.joblib -size +0c -print -quit 2>/dev/null | grep -q . && score=$((score+4)) || true
    if (( score > best_score )); then best=$p; best_score=$score; fi
  done
  if [[ -n "$best" ]]; then v4_abs_dir "$best"; return 0; fi
  mkdir -p "$V4_REPO_ROOT/outputs"; v4_abs_dir "$V4_REPO_ROOT/outputs"
}

V4_HOST_OUTPUTS=$(v4_choose_outputs_root)

V4_HOST_INPUT=${HOST_INPUT_DIR:-$(v4_first_existing_dir /data/voxel_csv_combined "$V4_REPO_ROOT/data/voxel_csv_combined" || true)}
if [[ -z "$V4_HOST_INPUT" ]]; then
  V4_HOST_INPUT=$(find /data "$V4_REPO_ROOT" -maxdepth 5 -type d -name voxel_csv_combined -print -quit 2>/dev/null || true)
fi
[[ -n "$V4_HOST_INPUT" && -d "$V4_HOST_INPUT" ]] || { echo "ERROR: could not auto-discover voxel_csv_combined" >&2; exit 2; }
V4_HOST_INPUT=$(v4_abs_dir "$V4_HOST_INPUT")

v4_build_session_inventory() {
  local out=$1 f rel geo parent sess seq
  : > "$out"
  while IFS= read -r -d '' f; do
    rel=${f#"$V4_HOST_INPUT"/}
    [[ "$rel" == */* ]] || continue
    geo=${rel%%/*}
    parent=$(basename "$(dirname "$f")")
    if [[ "$parent" =~ ^(session[0-9]+)_slice([0-9]+)$ ]]; then
      sess=${BASH_REMATCH[1]}; seq=$((10#${BASH_REMATCH[2]}))
      printf '%s/%s\t%d\t%s\n' "$geo" "$sess" "$seq" "$f" >> "$out"
    fi
  done < <(find "$V4_HOST_INPUT" -type f \( -name '*.csv' -o -name '*.csv.gz' \) -print0)
}

v4_discover_session() {
  if [[ -n ${SESSION_FILTER:-} ]]; then printf '%s\n' "$SESSION_FILTER"; return 0; fi
  local tmp groups g total unique dup chosen=''
  tmp=$(mktemp); groups=$(mktemp)
  v4_build_session_inventory "$tmp"
  [[ -s "$tmp" ]] || { echo "ERROR: no sessionN_sliceM CSV data below $V4_HOST_INPUT" >&2; rm -f "$tmp" "$groups"; return 2; }
  cut -f1 "$tmp" | sort -u > "$groups"
  # Largest valid session wins. A valid session has exactly one source per sequence.
  while read -r g; do
    [[ -n "$g" ]] || continue
    total=$(awk -F '\t' -v g="$g" '$1==g{n++}END{print n+0}' "$tmp")
    unique=$(awk -F '\t' -v g="$g" '$1==g{print $2}' "$tmp" | sort -n -u | wc -l | tr -d ' ')
    dup=$((total-unique))
    printf '%09d\t%09d\t%s\n' "$unique" "$dup" "$g"
  done < "$groups" | sort -k1,1nr -k2,2n -k3,3 | while IFS=$'\t' read -r unique dup g; do
    if [[ "$dup" == 000000000 ]]; then printf '%s\n' "$g"; break; fi
  done > "$groups.choice"
  chosen=$(head -1 "$groups.choice" 2>/dev/null || true); rm -f "$groups.choice"
  if [[ -z "$chosen" ]]; then
    echo "ERROR: every discovered session has duplicate CSV sources for at least one slice sequence." >&2
    echo "Inventory:" >&2; sort -k1,1 -k2,2n "$tmp" >&2
    rm -f "$tmp" "$groups"
    return 2
  fi
  rm -f "$tmp" "$groups"
  printf '%s\n' "$chosen"
}

V4_SESSION_FILTER=$(v4_discover_session)
[[ "$V4_SESSION_FILTER" == */session* ]] || { echo "ERROR: invalid discovered session: $V4_SESSION_FILTER" >&2; exit 2; }
V4_SESSION_SAFE=$(printf '%s' "$V4_SESSION_FILTER" | tr '/ ' '__')

V4_PROD_ROOT_HOST=${V4_PROD_ROOT_HOST:-$V4_HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_production}
V4_SESSION_ROOT_HOST=${V4_SESSION_ROOT_HOST:-$V4_PROD_ROOT_HOST/sessions/$V4_SESSION_SAFE}
V4_DIAG_ROOT_HOST=${V4_DIAG_ROOT_HOST:-$V4_PROD_ROOT_HOST/diagnostics}
V4_REVIEW_ROOT_HOST=${V4_REVIEW_ROOT_HOST:-$V4_PROD_ROOT_HOST/review_packages}
mkdir -p "$V4_SESSION_ROOT_HOST/backups" "$V4_DIAG_ROOT_HOST" "$V4_REVIEW_ROOT_HOST"

# Persist discovery inventory for debugging even if a later GPU test fails.
V4_SESSION_INVENTORY_HOST="$V4_SESSION_ROOT_HOST/session_inventory.tsv"
v4_build_session_inventory "$V4_SESSION_INVENTORY_HOST"

V4_MODEL_HOST=$(v4_first_existing_file "${MODEL_HOST:-$V4_HOST_OUTPUTS/poleline_voxel_run_session_groups/precision_v4/train/precision_best.pt}" 'precision_best.pt' "$V4_HOST_OUTPUTS" || true)
V4_CAL_HOST=$(v4_first_existing_file "${CAL_HOST:-$V4_HOST_OUTPUTS/poleline_voxel_run_session_groups/precision_v4/full_val/calibration.json}" 'calibration.json' "$V4_HOST_OUTPUTS" || true)
V4_STAGE2_HOST=$(v4_first_existing_file "${STAGE2_BUNDLE_HOST:-$V4_HOST_OUTPUTS/poleline_voxel_run_session_groups/v4_realtime/stage2_refiner/local_refiner_bundle.joblib}" 'local_refiner_bundle.joblib' "$V4_HOST_OUTPUTS" || true)
[[ -s "$V4_MODEL_HOST" ]] || { echo "ERROR: precision_best.pt not found under $V4_HOST_OUTPUTS" >&2; exit 2; }
[[ -s "$V4_CAL_HOST" ]] || { echo "ERROR: calibration.json not found under $V4_HOST_OUTPUTS" >&2; exit 2; }
[[ -s "$V4_STAGE2_HOST" ]] || { echo "ERROR: local_refiner_bundle.joblib not found under $V4_HOST_OUTPUTS" >&2; exit 2; }

V4_CODE_FINGERPRINT=$("$V4_HERE/v4_code_fingerprint.sh")
V4_MODEL_SHA256=$(sha256sum "$V4_MODEL_HOST" | awk '{print $1}')
V4_CAL_SHA256=$(sha256sum "$V4_CAL_HOST" | awk '{print $1}')
V4_STAGE2_SHA256=$(sha256sum "$V4_STAGE2_HOST" | awk '{print $1}')
V4_DEPLOY_FINGERPRINT=$(printf '%s\n%s\n%s\n%s\n' "$V4_CODE_FINGERPRINT" "$V4_MODEL_SHA256" "$V4_CAL_SHA256" "$V4_STAGE2_SHA256" | sha256sum | awk '{print $1}')
V4_DEPLOY_SHORT=${V4_DEPLOY_FINGERPRINT:0:20}

V4_RUN_ROOT_HOST=${V4_RUN_ROOT_HOST:-$V4_SESSION_ROOT_HOST/runs/$V4_DEPLOY_SHORT}
mkdir -p "$V4_RUN_ROOT_HOST"
printf '%s\n' "$V4_RUN_ROOT_HOST" > "$V4_SESSION_ROOT_HOST/LATEST_RUN_ROOT.txt"

v4_to_container_output() {
  local p=$1
  case "$p" in
    "$V4_HOST_OUTPUTS"/*) printf '/outputs/%s\n' "${p#"$V4_HOST_OUTPUTS"/}" ;;
    "$V4_HOST_OUTPUTS") printf '/outputs\n' ;;
    *) echo "ERROR: output path outside output root: $p" >&2; return 2 ;;
  esac
}
V4_RUN_ROOT=$(v4_to_container_output "$V4_RUN_ROOT_HOST")
V4_DIAG_ROOT=$(v4_to_container_output "$V4_DIAG_ROOT_HOST")
V4_REVIEW_ROOT=$(v4_to_container_output "$V4_REVIEW_ROOT_HOST")
V4_MODEL=$(v4_to_container_output "$V4_MODEL_HOST")
V4_CAL=$(v4_to_container_output "$V4_CAL_HOST")
V4_STAGE2=$(v4_to_container_output "$V4_STAGE2_HOST")

V4_RUNTIME_ENV_HOST=${RUNTIME_MODE_ENV:-$V4_DIAG_ROOT_HOST/v4_runtime_mode.env}
V4_GATED_FINGERPRINT_FILE=${V4_GATED_FINGERPRINT_FILE:-$V4_DIAG_ROOT_HOST/gated_deployment_sha256.txt}
V4_GATE_FINGERPRINT_MATCH=0
if [[ -s "$V4_RUNTIME_ENV_HOST" && -s "$V4_GATED_FINGERPRINT_FILE" && "$(cat "$V4_GATED_FINGERPRINT_FILE")" == "$V4_DEPLOY_FINGERPRINT" ]]; then source "$V4_RUNTIME_ENV_HOST"; V4_GATE_FINGERPRINT_MATCH=1; fi
V4_RUNTIME_MODE=${V4_RUNTIME_MODE:-full_cpu}
case "$V4_RUNTIME_MODE" in
  full_cpu) V4_EVALUATE_ALL_CORES=1; V4_GPU_COORD_CHANNELS=0; V4_FIXED_BATCH_SHAPE=0 ;;
  active_cpu) V4_EVALUATE_ALL_CORES=0; V4_GPU_COORD_CHANNELS=0; V4_FIXED_BATCH_SHAPE=1 ;;
  full_gpu) V4_EVALUATE_ALL_CORES=1; V4_GPU_COORD_CHANNELS=1; V4_FIXED_BATCH_SHAPE=0 ;;
  active_gpu) V4_EVALUATE_ALL_CORES=0; V4_GPU_COORD_CHANNELS=1; V4_FIXED_BATCH_SHAPE=1 ;;
  *) echo "ERROR: invalid V4_RUNTIME_MODE=$V4_RUNTIME_MODE" >&2; exit 2 ;;
esac
V4_BATCH_ENV_HOST=${BATCH_SIZE_ENV:-$V4_DIAG_ROOT_HOST/v4_batch_size.env}
if [[ "$V4_GATE_FINGERPRINT_MATCH" == 1 && -s "$V4_BATCH_ENV_HOST" ]]; then source "$V4_BATCH_ENV_HOST"; fi
V4_BATCH_SIZE=${V4_BATCH_SIZE:-12}
V4_ACCEPTANCE_FINGERPRINT_FILE=${V4_ACCEPTANCE_FINGERPRINT_FILE:-$V4_DIAG_ROOT_HOST/production_acceptance_deployment_sha256.txt}
V4_ACCEPTANCE_FINGERPRINT_MATCH=0
if [[ -s "$V4_ACCEPTANCE_FINGERPRINT_FILE" && "$(cat "$V4_ACCEPTANCE_FINGERPRINT_FILE")" == "$V4_DEPLOY_FINGERPRINT" ]]; then V4_ACCEPTANCE_FINGERPRINT_MATCH=1; fi

V4_CONTAINER_PREFIX=(
  "${V4_DOCKER[@]}" run --rm --gpus all
  -e PYTHONPYCACHEPREFIX=/tmp/pycache -e MPLCONFIGDIR=/tmp/matplotlib -e XDG_CACHE_HOME=/tmp/xdg-cache
  -e TORCH_HOME=/tmp/torch-home -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor -e TRITON_CACHE_DIR=/tmp/triton
  -e CUDA_CACHE_PATH=/tmp/cuda-cache -e NUMBA_CACHE_DIR=/tmp/numba-cache -e JOBLIB_TEMP_FOLDER=/tmp/joblib -e TMPDIR=/tmp
  --mount "type=bind,source=$V4_HERE,target=/workspace/v4,readonly"
  --mount "type=bind,source=$V4_HOST_OUTPUTS,target=/outputs"
  --mount "type=bind,source=$V4_HOST_INPUT,target=/data/voxel_csv_combined,readonly"
  --workdir /workspace/v4 "$V4_IMAGE"
)

v4_docker_run() {
  local cmd=$1
  "${V4_CONTAINER_PREFIX[@]}" bash -lc "set -euo pipefail; $cmd"
}

v4_write_context() {
  local f=$V4_RUN_ROOT_HOST/run_context.env tmp=$V4_RUN_ROOT_HOST/.run_context.$$.tmp
  {
    printf 'V4_CONTEXT_CREATED_UTC=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'V4_CODE_HOST=%q\n' "$V4_HERE"; printf 'V4_HOST_INPUT=%q\n' "$V4_HOST_INPUT"; printf 'V4_HOST_OUTPUTS=%q\n' "$V4_HOST_OUTPUTS"
    printf 'V4_SESSION_FILTER=%q\n' "$V4_SESSION_FILTER"; printf 'V4_SESSION_SAFE=%q\n' "$V4_SESSION_SAFE"; printf 'V4_SESSION_INVENTORY_HOST=%q\n' "$V4_SESSION_INVENTORY_HOST"
    printf 'V4_RUN_ROOT_HOST=%q\n' "$V4_RUN_ROOT_HOST"; printf 'V4_RUN_ROOT=%q\n' "$V4_RUN_ROOT"
    printf 'V4_MODEL_HOST=%q\n' "$V4_MODEL_HOST"; printf 'V4_CAL_HOST=%q\n' "$V4_CAL_HOST"; printf 'V4_STAGE2_HOST=%q\n' "$V4_STAGE2_HOST"
    printf 'V4_CODE_FINGERPRINT=%q\n' "$V4_CODE_FINGERPRINT"; printf 'V4_MODEL_SHA256=%q\n' "$V4_MODEL_SHA256"; printf 'V4_CAL_SHA256=%q\n' "$V4_CAL_SHA256"; printf 'V4_STAGE2_SHA256=%q\n' "$V4_STAGE2_SHA256"; printf 'V4_DEPLOY_FINGERPRINT=%q\n' "$V4_DEPLOY_FINGERPRINT"
    printf 'V4_RUNTIME_MODE=%q\n' "$V4_RUNTIME_MODE"; printf 'V4_BATCH_SIZE=%q\n' "$V4_BATCH_SIZE"; printf 'V4_GATE_FINGERPRINT_MATCH=%q\n' "$V4_GATE_FINGERPRINT_MATCH"; printf 'V4_ACCEPTANCE_FINGERPRINT_MATCH=%q\n' "$V4_ACCEPTANCE_FINGERPRINT_MATCH"
    printf 'V4_EVALUATE_ALL_CORES=%q\n' "$V4_EVALUATE_ALL_CORES"; printf 'V4_GPU_COORD_CHANNELS=%q\n' "$V4_GPU_COORD_CHANNELS"; printf 'V4_FIXED_BATCH_SHAPE=%q\n' "$V4_FIXED_BATCH_SHAPE"; printf 'V4_IMAGE=%q\n' "$V4_IMAGE"
  } > "$tmp"; mv "$tmp" "$f"
  cp "$f" "$V4_SESSION_ROOT_HOST/run_context.env"
}
v4_write_context

v4_print_context() {
  cat <<CTX
V4 code:              $V4_HERE
Input root:           $V4_HOST_INPUT
Output root:          $V4_HOST_OUTPUTS
Selected session:     $V4_SESSION_FILTER
Session inventory:    $V4_SESSION_INVENTORY_HOST
Deployment:           $V4_DEPLOY_SHORT
Persistent run:       $V4_RUN_ROOT_HOST
Runtime mode:         $V4_RUNTIME_MODE
Batch size:           $V4_BATCH_SIZE
Runtime gate current: $V4_GATE_FINGERPRINT_MATCH
Acceptance current:   $V4_ACCEPTANCE_FINGERPRINT_MATCH
Model:                $V4_MODEL_HOST
Calibration:          $V4_CAL_HOST
Stage2 bundle:        $V4_STAGE2_HOST
Context file:         $V4_RUN_ROOT_HOST/run_context.env
CTX
}

v4_backup_code() {
  local stamp backup
  stamp=$(date -u +%Y%m%dT%H%M%SZ); backup="$V4_SESSION_ROOT_HOST/backups/v4_code_${V4_DEPLOY_SHORT}_${stamp}.tar.gz"
  tar -C "$V4_HERE" --exclude='__pycache__' --exclude='*.pyc' -czf "$backup" .
  printf '%s\n' "$backup" > "$V4_SESSION_ROOT_HOST/backups/LATEST_CODE_BACKUP.txt"; printf '%s\n' "$backup"
}

v4_detached_run() {
  local name=$1 log_container=$2 cmd=$3 state_file=$V4_RUN_ROOT_HOST/current_container.txt
  if "${V4_DOCKER[@]}" inspect "$name" >/dev/null 2>&1; then
    local running; running=$("${V4_DOCKER[@]}" inspect -f '{{.State.Running}}' "$name" 2>/dev/null || printf false)
    if [[ "$running" == true ]]; then echo "Container already running: $name"; printf '%s\n' "$name" > "$state_file"; return 0; fi
    "${V4_DOCKER[@]}" rm "$name" >/dev/null
  fi
  local id
  id=$("${V4_DOCKER[@]}" run -d --name "$name" --gpus all \
    -e PYTHONPYCACHEPREFIX=/tmp/pycache -e MPLCONFIGDIR=/tmp/matplotlib -e XDG_CACHE_HOME=/tmp/xdg-cache \
    -e TORCH_HOME=/tmp/torch-home -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor -e TRITON_CACHE_DIR=/tmp/triton \
    -e CUDA_CACHE_PATH=/tmp/cuda-cache -e NUMBA_CACHE_DIR=/tmp/numba-cache -e JOBLIB_TEMP_FOLDER=/tmp/joblib -e TMPDIR=/tmp \
    --mount "type=bind,source=$V4_HERE,target=/workspace/v4,readonly" --mount "type=bind,source=$V4_HOST_OUTPUTS,target=/outputs" --mount "type=bind,source=$V4_HOST_INPUT,target=/data/voxel_csv_combined,readonly" \
    --workdir /workspace/v4 "$V4_IMAGE" bash -lc "set -euo pipefail; mkdir -p \"\$(dirname '$log_container')\"; { $cmd; } 2>&1 | tee '$log_container'")
  printf '%s\n' "$name" > "$state_file"; printf '%s\n' "$id" > "$V4_RUN_ROOT_HOST/current_container_id.txt"
  echo "Started detached container: $name ($id)"; echo "Persistent output: $V4_RUN_ROOT_HOST"; echo "Check: $V4_HERE/show_v4_production_state_on_nebius.sh"
}

v4_latest_test_root_host() {
  local p=''
  p=$(cat "$V4_RUN_ROOT_HOST/LATEST_TEST_ROOT.txt" 2>/dev/null || true)
  [[ -n "$p" && -d "$p" ]] || p=$(cat "$V4_SESSION_ROOT_HOST/LATEST_TEST_ROOT.txt" 2>/dev/null || true)
  [[ -n "$p" && -d "$p" ]] || return 1
  printf '%s\n' "$p"
}

v4_require_current_gate() {
  if [[ "$V4_GATE_FINGERPRINT_MATCH" != 1 ]]; then
    if [[ "${ALLOW_UNGATED_REFERENCE:-0}" == 1 ]]; then echo "WARNING: ungated reference runtime explicitly allowed" >&2; return 0; fi
    echo "ERROR: this exact code/model/calibration/Stage2 deployment has not passed the H100 runtime gate." >&2
    echo "Run: $V4_HERE/run_v4_runtime_variant_gate_on_nebius.sh" >&2; exit 3
  fi
}

v4_require_acceptance() {
  v4_require_current_gate
  if [[ "$V4_ACCEPTANCE_FINGERPRINT_MATCH" != 1 ]]; then echo "ERROR: this exact deployment has not passed full production acceptance." >&2; echo "Run: $V4_HERE/run_v4_production_tests_on_nebius.sh" >&2; exit 4; fi
}
