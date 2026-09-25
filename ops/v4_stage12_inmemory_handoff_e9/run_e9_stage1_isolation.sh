#!/usr/bin/env bash
# E9 representative-session Stage 1 isolation. It never edits production data.
# Invoke as "bash ... --run" so a nonzero result cannot log out the SSH shell.

e9_isolation_main() {
  local mode=${1:---monitor}
  local repo=${REPO:-/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10}
  local outputs=${HOST_OUTPUTS:-/workspace/voxel_poleline/outputs}
  local image=${IMAGE:-va-v4-realtime:torch241-cu121}
  local sid=NYSEGDistVegMgnt_AUBURN_-_SPRUCE_HAVEN_FARMS_TAP_520-3p_2029__session2
  local gid=NYSEGDistVegMgnt_AUBURN_-_SPRUCE_HAVEN_FARMS_TAP_520-3p_2029/session2
  local baseline="$outputs/poleline_voxel_run_session_groups/v4_production/full_dataset_runs/d9977c39c443f5fa14f8/20260825T203403Z/stage1/$sid"
  local calibration="$outputs/poleline_voxel_run_session_groups/precision_v4/full_val/calibration.json"
  local model="$outputs/poleline_voxel_run_session_groups/precision_v4/train/precision_best.pt"
  local ops="$repo/ops/v4_stage1_inference_opt2"
  local e9="$repo/ops/v4_stage12_inmemory_handoff_e9"
  local stamp=${E9_ISOLATION_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
  local root=${E9_ISOLATION_ROOT:-$outputs/v4_stage12_e9_stage1_isolation/$stamp}
  local label graph pid run log activity now state count report rc accepted failed=0

  if [ "$mode" = --monitor ]; then
    root=${2:-$(cat /home/agni/LATEST_E9_S1_ISOLATION_ROOT.txt 2>/dev/null)}
    printf 'ISOLATION_ROOT=%s\n' "$root"
    cat "$root/STATE.txt" 2>/dev/null || true
    cat "$root/OUTCOMES.tsv" 2>/dev/null || true
    if [ -s "$root/CURRENT_LOG.txt" ]; then
      log=$(cat "$root/CURRENT_LOG.txt")
      tail -n 45 "$log" 2>/dev/null || true
    fi
    return 0
  fi
  if [ "$mode" != --run ]; then printf 'USAGE: bash %s --run|--monitor [ROOT]\n' "$0"; return 2; fi
  for file in "$baseline" "$model" "$calibration" "$ops/launch_v4_stage1_opt2_experiment.sh" "$e9/diagnose_e9_stage1_repro.py"; do
    if [ ! -e "$file" ]; then printf 'MISSING=%s\n' "$file"; return 2; fi
  done
  count=$(find "$baseline" -name '*_stage1.npz' -type f 2>/dev/null | wc -l)
  if [ "$count" -ne 157 ]; then printf 'EXPECTED_157_PRODUCTION_NPZ_FOUND=%s\n' "$count"; return 2; fi
  if [ "$(sha256sum "$model" | cut -d' ' -f1)" != 1b8b20c0bb2b52a1617555ed72c34311ba3839effd674bb2cac5273040d909ee ]; then
    printf 'MODEL_SHA256_MISMATCH\n'; return 2
  fi
  if [ "$(sha256sum "$calibration" | cut -d' ' -f1)" != dea4829143f33d1f674176185ecd59df620c50a70488a83b0a2d6e17b81784e1 ]; then
    printf 'CALIBRATION_SHA256_MISMATCH\n'; return 2
  fi
  mkdir -p "$root/reports" "$root/logs" "$root/metadata"
  printf '%s\n' "$root" > /home/agni/LATEST_E9_S1_ISOLATION_ROOT.txt
  printf 'label\tgraph\tproduction_exact\treport\trun_root\n' > "$root/OUTCOMES.tsv"
  printf 'STATE=RUNNING\n' > "$root/STATE.txt"
  git -C "$repo" rev-parse HEAD > "$root/metadata/source_commit.txt" 2>&1 || true
  sha256sum "$repo/v4/v4_realtime_core.py" "$ops/v4_realtime_core_opt2.py" \
    "$ops/run_v4_stage1_opt2.py" "$ops/compare_v4_stage1_opt2_quality.py" \
    "$model" "$calibration" > "$root/metadata/source_and_assets_sha256.txt" 2>&1 || true
  for prior in 20260924T202241Z_e9_s1_fidelity_probe 20260924T225930Z_e9_control_r1 \
    20260924T230931Z_e9_control_r2 20260924T182337Z_e9_control_full30; do
    file="$outputs/poleline_voxel_run_session_groups/v4_production/stage1_opt2_experiments/$prior/RUN_INFO.txt"
    if [ -s "$file" ]; then cp "$file" "$root/metadata/$prior.RUN_INFO.txt"; fi
  done

  # A-B-A-B, with fresh roots and identical flags except CUDA Graph.
  for label in e9_isolate_graph0_r1 e9_isolate_graph1_r1 e9_isolate_graph0_r2 e9_isolate_graph1_r2; do
    case "$label" in *graph1*) graph=1;; *) graph=0;; esac
    local run_stamp
    run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
    run="$outputs/poleline_voxel_run_session_groups/v4_production/stage1_opt2_experiments/${run_stamp}_${label}"
    log="$root/logs/$label.launch.log"
    printf '%s\n' "$log" > "$root/CURRENT_LOG.txt"
    printf 'STATE=RUNNING label=%s graph=%s run=%s\n' "$label" "$graph" "$run" > "$root/STATE.txt"
    RUN_STAMP="$run_stamp" VARIANT_NAME="$label" EXPECTED_SESSIONS=1 ONLY_GROUP_ID="$gid" \
      IMAGE="$image" RESUME=0 SCORE_ATOL=0 DETAILED_CUDA_TIMING=0 \
      COMPILE_MODEL=0 COMPILE_MODE=default BATCH_SIZE=12 CHANNELS_LAST=1 PINNED_D2H=0 \
      RETAIN_GATHER_HOST_BUFFERS=1 CACHE_REFERENCE_COORDINATE_CHANNELS=1 USE_CUDA_GRAPH="$graph" \
      PREFETCH_INPUTS=0 PREFETCH_DEPTH=1 PREFETCH_WORKERS=1 PREPARE_CORE_SCHEDULE=0 \
      ASYNC_OUTPUT_WRITES=0 GROUPNORM_INPUT_LAYOUT=0 CONV_INPUT_LAYOUT=0 \
      CHANNELS_LAST_WEIGHTS=0 KERNEL_FACTORY_INPUT_PACK=0 \
      bash "$ops/launch_v4_stage1_opt2_experiment.sh" > "$log" 2>&1
    rc=$?
    pid=$(sed -n 's/^PID=//p' "$log" | tail -n 1)
    if [ "$rc" -ne 0 ] || [ -z "$pid" ]; then
      printf 'LAUNCH_FAILED label=%s rc=%s\n' "$label" "$rc" >> "$root/metadata/failures.txt"
      failed=$((failed+1))
      continue
    fi
    activity=$(date +%s)
    while :; do
      if [ -s "$run/STAGE1_OPT2_PRODUCTION_EQUIVALENT_COMPLETE.txt" ]; then break; fi
      if find "$run/status" -maxdepth 1 -name '*.failed' -type f -print -quit 2>/dev/null | grep -q .; then break; fi
      if ! kill -0 "$pid" 2>/dev/null; then break; fi
      state=$(ps -o stat= -p "$pid" 2>/dev/null || true)
      case "$state" in Z*|'') break;; esac
      now=$(find "$run/status" "$run/logs/stage1" "$run/logs/stage1_export" \
        -maxdepth 1 -type f -printf '%T@\n' 2>/dev/null | sort -nr | head -n 1 | cut -d. -f1)
      if [ -n "$now" ] && [ "$now" -gt "$activity" ]; then activity=$now; fi
      now=$(date +%s)
      if [ $((now-activity)) -gt 600 ]; then
        printf 'STALL label=%s idle_seconds=%s pid=%s\n' "$label" "$((now-activity))" "$pid" >> "$root/metadata/failures.txt"
        kill -TERM "$pid" 2>/dev/null || true
        break
      fi
      sleep 10
    done
    accepted=NO
    if [ -s "$run/STAGE1_OPT2_PRODUCTION_EQUIVALENT_COMPLETE.txt" ]; then accepted=YES; fi
    cp "$run/RUN_INFO.txt" "$root/metadata/$label.RUN_INFO.txt" 2>/dev/null || true
    cp "$run/STAGE1_OPT2_DRIVER.log" "$root/logs/$label.driver.log" 2>/dev/null || true
    report="$root/reports/$label.json"
    docker run --rm \
      --mount "type=bind,source=$outputs,target=/outputs,readonly" \
      --mount "type=bind,source=$repo/v4,target=/workspace/v4,readonly" \
      --mount "type=bind,source=$e9,target=/workspace/e9,readonly" \
      --mount "type=bind,source=$root/reports,target=/reports" \
      -e PYTHONPATH=/workspace/v4 "$image" python /workspace/e9/diagnose_e9_stage1_repro.py \
      --reference "/outputs${baseline#$outputs}" \
      --run "$label=/outputs${run#$outputs}/stage1/$sid" \
      --calibration "/outputs${calibration#$outputs}" \
      --report "/reports/$label.json" >> "$log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ] || [ ! -s "$report" ]; then accepted=NO; fi
    if [ "$accepted" = YES ] && grep -q '"exact_stage1_payload": true' "$report"; then
      printf '%s\t%s\tPASS\t%s\t%s\n' "$label" "$graph" "$report" "$run" >> "$root/OUTCOMES.tsv"
    else
      printf '%s\t%s\tFAIL\t%s\t%s\n' "$label" "$graph" "$report" "$run" >> "$root/OUTCOMES.tsv"
      failed=$((failed+1))
    fi
  done
  if [ "$failed" -eq 0 ]; then
    printf 'STATE=ALL_FOUR_EXACT; E9_FULL30_NOT_RUN\n' > "$root/STATE.txt"
  else
    printf 'STATE=STAGE1_REPRO_FAILURE; FAILED_RUNS=%s; E9_FULL30_NOT_RUN\n' "$failed" > "$root/STATE.txt"
  fi
  rm -f "$root/CURRENT_LOG.txt"
  local archive="/home/agni/e9_stage1_isolation_${stamp}.tar.gz"
  tar -czf "$archive" -C "$root" STATE.txt OUTCOMES.tsv metadata reports logs
  sha256sum "$archive" > "$archive.sha256"
  printf '%s\n' "$archive" > /home/agni/LATEST_E9_S1_ISOLATION_ARCHIVE.txt
  cat "$root/STATE.txt"
  printf 'UPLOAD_ARCHIVE=%s\n' "$archive"
  return 0
}
e9_isolation_main "$@"
