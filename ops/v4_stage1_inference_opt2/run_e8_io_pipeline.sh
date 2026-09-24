#!/usr/bin/env bash
# E8 control/candidate/full-gate harness. Run as an executable script, never source it.

main() {
  mode=${1:---run}
  repo=${REPO:-/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10}
  ops="$repo/ops/v4_stage1_inference_opt2"
  outputs=${HOST_OUTPUTS:-/workspace/voxel_poleline/outputs}
  image=${IMAGE:-va-v4-realtime:torch241-cu121}
  group=${E8_GROUP_ID:-NYSEGDistVegMgnt_AUBURN_-_SPRUCE_HAVEN_FARMS_TAP_520-3p_2029/session2}

  if [ "$mode" = "--monitor" ]; then
    root=${2:-$(cat /home/agni/LATEST_V4_STAGE1_E8_IO_HARNESS.txt 2>/dev/null)}
    if [ -z "$root" ]; then
      echo "E8_STATUS=NO_HARNESS_RECORDED"
      return
    fi
    echo "E8_HARNESS_ROOT=$root"
    cat "$root/STATE.txt" 2>/dev/null || true
    echo "===== RECENT LOG ====="
    tail -n 100 "$root/E8_HARNESS.log" 2>/dev/null || true
    echo "===== ROOTS ====="
    find "$root/roots" -maxdepth 1 -type f -print -exec sh -c 'printf "  "; cat "$1"' _ {} \; 2>/dev/null || true
    return
  fi

  if [ ! -d "$ops" ]; then
    echo "E8_STATUS=STOP_MISSING_OPS"
    return
  fi

  stamp=${E8_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
  harness="$outputs/v4_stage1_e8_io_pipeline/$stamp"
  mkdir -p "$harness"/{roots,summaries}
  printf '%s\n' "$harness" > /home/agni/LATEST_V4_STAGE1_E8_IO_HARNESS.txt
  printf '%s\n' "E8_STATUS=RUNNING created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$harness/STATE.txt"
  exec > >(tee -a "$harness/E8_HARNESS.log") 2>&1

  echo "E8_HARNESS_ROOT=$harness"
  echo "E8_GROUP_ID=$group"
  echo "E8_UPDATE_DATE=2026-09-24"

  host_to_container() {
    case "$1" in
      "$outputs") printf '/outputs\n' ;;
      "$outputs"/*) printf '/outputs%s\n' "${1#$outputs}" ;;
      *) printf '%s\n' "" ;;
    esac
  }

  wait_for_run() {
    label=$1
    pid=$(cat /home/agni/LATEST_V4_STAGE1_OPT2_PID.txt 2>/dev/null)
    run_root=$(cat /home/agni/LATEST_V4_STAGE1_OPT2_RUN.txt 2>/dev/null)
    log=$(cat /home/agni/LATEST_V4_STAGE1_OPT2_LAUNCH_LOG.txt 2>/dev/null)
    printf '%s\n' "$run_root" > "$harness/roots/$label.txt"
    while [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; do
      printf '%s\n' "E8_STATUS=RUNNING label=$label pid=$pid" > "$harness/STATE.txt"
      sleep 20
    done
    if [ -s "$run_root/STAGE1_OPT2_PRODUCTION_EQUIVALENT_COMPLETE.txt" ]; then
      echo "E8_VARIANT_ACCEPTED label=$label root=$run_root"
      return 0
    fi
    echo "E8_VARIANT_REJECTED label=$label root=$run_root log=$log"
    tail -n 100 "$log" 2>/dev/null || true
    return 1
  }

  summarize_run() {
    label=$1
    run_root=$(cat "$harness/roots/$label.txt")
    run_c=$(host_to_container "$run_root")
    out_c=$(host_to_container "$harness/summaries/$label.json")
    docker run --rm \
      --mount "type=bind,source=$ops,target=/workspace/opt,readonly" \
      --mount "type=bind,source=$outputs,target=/outputs" \
      "$image" python /workspace/opt/summarize_e8_pipeline.py \
        --run-root "$run_c" --exclude-first 1 --output "$out_c"
  }

  launch_variant() {
    label=$1
    pipeline=$2
    RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ) \
    VARIANT_NAME="$label" \
    EXPECTED_SESSIONS=1 \
    ONLY_GROUP_ID="$group" \
    IMAGE="$image" \
    DETAILED_CUDA_TIMING=0 \
    RETAIN_GATHER_HOST_BUFFERS=1 \
    CACHE_REFERENCE_COORDINATE_CHANNELS=1 \
    CACHE_COORDINATE_CHANNELS=0 \
    PRECOMPUTE_BATCH_GATHER_PLANS=0 \
    USE_CUDA_GRAPH=1 \
    PREFETCH_INPUTS="$pipeline" \
    PREPARE_CORE_SCHEDULE=0 \
    PREFETCH_DEPTH=4 \
    PREFETCH_WORKERS=2 \
    ASYNC_OUTPUT_WRITES="$pipeline" \
    GROUPNORM_INPUT_LAYOUT=0 \
    CONV_INPUT_LAYOUT=0 \
    CHANNELS_LAST_WEIGHTS=0 \
    KERNEL_FACTORY_INPUT_PACK=0 \
    SCORE_ATOL=0 \
    RESUME=1 \
    bash "$ops/launch_v4_stage1_opt2_experiment.sh"
    wait_for_run "$label" || return
    summarize_run "$label"
  }

  if [ "$mode" = "--run" ]; then
    launch_variant e7_e8_control_r1 0 || { printf '%s\n' "E8_STATUS=REJECTED_CONTROL_R1" > "$harness/STATE.txt"; return; }
    launch_variant e8_io_pipeline_r1 1 || { printf '%s\n' "E8_STATUS=REJECTED_E8_R1" > "$harness/STATE.txt"; return; }
    launch_variant e7_e8_control_r2 0 || { printf '%s\n' "E8_STATUS=REJECTED_CONTROL_R2" > "$harness/STATE.txt"; return; }
    launch_variant e8_io_pipeline_r2 1 || { printf '%s\n' "E8_STATUS=REJECTED_E8_R2" > "$harness/STATE.txt"; return; }

    hc=$(host_to_container "$harness")
    docker run --rm \
      --mount "type=bind,source=$ops,target=/workspace/opt,readonly" \
      --mount "type=bind,source=$outputs,target=/outputs" \
      "$image" python /workspace/opt/gate_e8_pipeline.py \
        --control "$hc/summaries/e7_e8_control_r1.json" \
        --control "$hc/summaries/e7_e8_control_r2.json" \
        --candidate "$hc/summaries/e8_io_pipeline_r1.json" \
        --candidate "$hc/summaries/e8_io_pipeline_r2.json" \
        --minimum-save-percent 1.0 \
        --output "$hc/E8_PAIR_GATE.json"

    gate_status=$(sed -n 's/^[[:space:]]*"status": "\([^"]*\)".*/\1/p' "$harness/E8_PAIR_GATE.json")
    if [ "$gate_status" != "PASS" ]; then
      printf '%s\n' "E8_STATUS=$gate_status" > "$harness/STATE.txt"
      cat "$harness/E8_PAIR_GATE.json"
      return
    fi

    echo "E8_PAIR_GATE=PASS; launching full 30-session E8 candidate."
    RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ) \
    VARIANT_NAME=e8_io_pipeline_full30 \
    EXPECTED_SESSIONS=30 \
    ONLY_GROUP_ID= \
    IMAGE="$image" \
    DETAILED_CUDA_TIMING=0 \
    RETAIN_GATHER_HOST_BUFFERS=1 \
    CACHE_REFERENCE_COORDINATE_CHANNELS=1 \
    CACHE_COORDINATE_CHANNELS=0 \
    PRECOMPUTE_BATCH_GATHER_PLANS=0 \
    USE_CUDA_GRAPH=1 \
    PREFETCH_INPUTS=1 \
    PREPARE_CORE_SCHEDULE=0 \
    PREFETCH_DEPTH=4 \
    PREFETCH_WORKERS=2 \
    ASYNC_OUTPUT_WRITES=1 \
    GROUPNORM_INPUT_LAYOUT=0 \
    CONV_INPUT_LAYOUT=0 \
    CHANNELS_LAST_WEIGHTS=0 \
    KERNEL_FACTORY_INPUT_PACK=0 \
    SCORE_ATOL=0 \
    RESUME=1 \
    bash "$ops/launch_v4_stage1_opt2_experiment.sh"
    wait_for_run e8_io_pipeline_full30 || { printf '%s\n' "E8_STATUS=REJECTED_FULL30" > "$harness/STATE.txt"; return; }
    summarize_run e8_io_pipeline_full30

    full_root=$(cat "$harness/roots/e8_io_pipeline_full30.txt")
    printf '%s\n' "# E8 result" "" \
      "- Update date (UTC): $(date -u +%Y-%m-%d)" \
      "- Result created (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "- Contract: E7 plus bounded CPU prefetch/prepare and one ordered async writer only." \
      "- Full run: 30 accepted sessions, zero failures, 30 saved-production equivalence reports." \
      "- Full root: $full_root" "" \
      "## Paired timing gate" "" "~~~~json" > "$harness/E8_RESULT.md"
    cat "$harness/E8_PAIR_GATE.json" >> "$harness/E8_RESULT.md"
    printf '%s\n' "~~~~" "" "## Full timing summary" "" "~~~~json" >> "$harness/E8_RESULT.md"
    cat "$harness/summaries/e8_io_pipeline_full30.json" >> "$harness/E8_RESULT.md"
    printf '%s\n' "~~~~" >> "$harness/E8_RESULT.md"

    archive="/home/agni/v4_stage1_e8_io_pipeline_${stamp}.tar.gz"
    tar -czf "$archive.partial" -C "$harness" E8_RESULT.md E8_PAIR_GATE.json summaries roots STATE.txt E8_HARNESS.log
    mv "$archive.partial" "$archive"
    sha256sum "$archive" > "$archive.sha256"
    printf '%s\n' "$archive" > /home/agni/LATEST_V4_STAGE1_E8_IO_RESULT_ARCHIVE.txt
    printf '%s\n' "E8_STATUS=ACCEPTED_FULL30" > "$harness/STATE.txt"
    echo "E8_ACCEPTED_FULL30"
    echo "E8_RESULT_ARCHIVE=$archive"
    return
  fi

  echo "E8_STATUS=STOP_UNKNOWN_ARGUMENT mode=$mode"
}

main "$@"
