#!/usr/bin/env bash
# E9 full-dataset controller: disk round-trip control vs in-memory Stage1->Stage2.
# It returns from failures instead of exiting an interactive shell.

stage1_activity_epoch() {
  local runroot=$1
  find "$runroot/status" "$runroot/logs/stage1" "$runroot/logs/stage1_export" \
    -maxdepth 1 -type f \( -name '*.progress.json' -o -name '*.log' \
    -o -name '*.stage1.ok' -o -name '*.production_equivalence.json' \) \
    -printf '%T@\n' 2>/dev/null | sort -nr | head -n 1 | cut -d. -f1
}

main() {
  mode=${1:---run}
  repo=${REPO:-/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10}
  outputs=${HOST_OUTPUTS:-/workspace/voxel_poleline/outputs}
  input=${HOST_INPUT:-/data/voxel_csv_combined}
  image=${IMAGE:-va-v4-realtime:torch241-cu121}
  baseline=${BASELINE_RUN_HOST:-$outputs/poleline_voxel_run_session_groups/v4_production/full_dataset_runs/d9977c39c443f5fa14f8/20260825T203403Z}
  model=${MODEL_HOST:-$outputs/poleline_voxel_run_session_groups/precision_v4/train/precision_best.pt}
  calibration=${CALIBRATION_HOST:-$outputs/poleline_voxel_run_session_groups/precision_v4/full_val/calibration.json}
  s1_ops="$repo/ops/v4_stage1_inference_opt2"
  s2_ops="$repo/ops/v4_stage2_stage1_electrical_v10"
  e9_ops="$repo/ops/v4_stage12_inmemory_handoff_e9"
  bundle=${STAGE2_BUNDLE:-$outputs/poleline_voxel_run_session_groups/v4_realtime/stage2_refiner/local_refiner_bundle.joblib}
  profile=${STAGE2_PROFILE:-$outputs/poleline_voxel_run_session_groups/v4_stage23_quality/full_run_v2_20260904T194258Z/selection/selected_electrical_profile.json}
  group=${E9_GROUP_ID:-NYSEGDistVegMgnt_AUBURN_-_SPRUCE_HAVEN_FARMS_TAP_520-3p_2029/session2}
  stall_seconds=${E9_STALL_SECONDS:-600}
  max_attempts=${E9_MAX_ATTEMPTS:-2}
  session_timeout=${E9_SESSION_TIMEOUT_SECONDS:-7200}
  full30_control_stamp=${E9_FULL30_CONTROL_STAMP:-}

  if [ "$mode" = "--monitor" ]; then
    root=${2:-$(cat /home/agni/LATEST_V4_STAGE12_E9_HARNESS.txt 2>/dev/null)}
    echo "E9_HARNESS_ROOT=$root"
    cat "$root/STATE.txt" 2>/dev/null || true
    echo "===== RECENT HARNESS LOG ====="
    tail -n 120 "$root/E9_HARNESS.log" 2>/dev/null || true
    echo "===== DIAGNOSTICS ====="
    find "$root/diagnostics" -maxdepth 1 -type f -printf '%f\n' 2>/dev/null | sort || true
    return
  fi
  if [ "$mode" = "--self-test" ]; then
    if ! docker run --rm \
      --mount "type=bind,source=$repo/v4,target=/workspace/v4,readonly" \
      --mount "type=bind,source=$e9_ops,target=/workspace/e9,readonly" \
      --workdir /workspace/v4 -e PYTHONPATH=/workspace/v4:/workspace/e9 \
      "$image" python /workspace/e9/self_test_e9_stage12.py; then
      return 1
    fi
    local probe probe_epoch stale_epoch
    probe=$(mktemp -d)
    mkdir -p "$probe/status" "$probe/logs/stage1" "$probe/logs/stage1_export"
    touch -d '20 minutes ago' "$probe/logs/stage1/quiet.log"
    touch -d '20 minutes ago' "$probe/status/sample.progress.json"
    stale_epoch=$(stage1_activity_epoch "$probe")
    touch "$probe/status/sample.progress.json"
    probe_epoch=$(stage1_activity_epoch "$probe")
    rm -rf "$probe"
    if [ -z "$stale_epoch" ] || [ $(( $(date +%s) - stale_epoch )) -lt 600 ] \
      || [ -z "$probe_epoch" ] || [ $(( $(date +%s) - probe_epoch )) -gt 60 ]; then
      echo 'E9_WATCHDOG_SELF_TEST_FAILED'
      return 1
    fi
    echo 'E9_WATCHDOG_SELF_TEST_OK'
    return
  fi
  # E9 fidelity controls must start from fresh roots. Reusing a prior full-30
  # control could preserve an earlier, non-equivalent Stage-1 payload.
  if [ -n "$full30_control_stamp" ]; then
    echo 'E9_STATUS=STOP_PRIOR_FULL30_CONTROL_DISALLOWED; start a fresh run without E9_FULL30_CONTROL_STAMP'
    return
  fi
  for path in "$repo/.git" "$s1_ops/run_v4_stage1_opt2.py" "$s2_ops/run_v4_stage2_stage1_electrical_tracks.py" "$e9_ops/run_e9_stage12_inmemory.py" "$baseline/PHASE1_STAGE1_OK.txt" "$input" "$model" "$calibration" "$bundle" "$profile"; do
    if [ ! -e "$path" ]; then
      echo "E9_STATUS=STOP_MISSING_REQUIRED_PATH path=$path"
      return
    fi
  done
  for asset in \
    "$model:1b8b20c0bb2b52a1617555ed72c34311ba3839effd674bb2cac5273040d909ee" \
    "$calibration:dea4829143f33d1f674176185ecd59df620c50a70488a83b0a2d6e17b81784e1" \
    "$bundle:c451d501c3a3ccf7598ce8254d4807483afe3f41e4500be8a9abcf705843e72d" \
    "$profile:de79c637e9d70ff0d39c2765b8b6514f08a8547374079cd1f9f803cb9879ca1d"; do
    asset_path=${asset%:*}; required_sha=${asset##*:}
    actual_sha=$(sha256sum "$asset_path" | awk '{print $1}')
    if [ "$actual_sha" != "$required_sha" ]; then
      echo "E9_STATUS=STOP_ASSET_SHA_MISMATCH path=$asset_path expected=$required_sha actual=$actual_sha"
      return
    fi
  done

  stamp=${E9_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
  root="$outputs/v4_stage12_inmemory_handoff_e9/$stamp"
  mkdir -p "$root"/{control,candidate,comparisons,diagnostics,status,summaries}
  printf '%s\n' "$root" > /home/agni/LATEST_V4_STAGE12_E9_HARNESS.txt
  printf '%s\n' "E9_STATUS=RUNNING created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$root/STATE.txt"
  exec > >(tee -a "$root/E9_HARNESS.log") 2>&1
  echo "E9_HARNESS_ROOT=$root"
  echo "E9_CONTRACT=exact_stage1_payload;all_stage2_outputs;serial_refiner;no_candidate_stage1_artifacts"
  echo "E9_STAGE1_CONFIG=use_cuda_graph=0;resume=0;score_atol=0;both_arms"

  package_failure() {
    if [ -s "$root/summaries/E9_RESULT.json" ]; then return; fi
    mkdir -p "$root/summaries" "$root/comparisons" "$root/diagnostics"
    printf '%s\n' '{"decision":"NOT_ACCEPTED","status":"INCOMPLETE","reason":"The full validation did not complete. Review STATE.txt, E9_HARNESS.log and diagnostics before a controlled restart."}' > "$root/summaries/E9_RESULT.json"
    {
      echo '# E9 in-memory handoff: incomplete run'
      echo
      echo '**Decision: NOT_ACCEPTED (validation incomplete).**'
      echo
      echo 'No performance or exact-output acceptance can be inferred from a partial run.'
      echo
      echo '## Last recorded state'
      echo
      cat "$root/STATE.txt"
      echo
      echo 'The diagnostic archive contains the logs and comparison reports available at failure.'
    } > "$root/summaries/E9_RESULT.md"
    failure_archive="/home/agni/v4_stage12_e9_${stamp}_incomplete.tar.gz"
    tar -czf "$failure_archive.partial" -C "$root" STATE.txt E9_HARNESS.log summaries comparisons diagnostics 2>/dev/null && {
      mv "$failure_archive.partial" "$failure_archive"
      sha256sum "$failure_archive" > "$failure_archive.sha256"
      printf '%s\n' "$failure_archive" > /home/agni/LATEST_V4_STAGE12_E9_RESULT_ARCHIVE.txt
      echo "E9_DIAGNOSTIC_ARCHIVE=$failure_archive"
    }
  }
  trap package_failure EXIT
  if ! docker run --rm \
    --mount "type=bind,source=$repo/v4,target=/workspace/v4,readonly" \
    --mount "type=bind,source=$e9_ops,target=/workspace/e9,readonly" \
    --workdir /workspace/v4 -e PYTHONPATH=/workspace/v4:/workspace/e9 \
    "$image" python /workspace/e9/self_test_e9_stage12.py; then
    printf '%s\n' 'E9_STATUS=STOP_SELF_TEST_FAILED' > "$root/STATE.txt"
    return
  fi

  host_to_container() {
    case "$1" in
      "$outputs") printf '/outputs\n' ;;
      "$outputs"/*) printf '/outputs%s\n' "${1#$outputs}" ;;
      *) printf '\n' ;;
    esac
  }
  gid_from_manifest() {
    awk -F',' 'NR==1 {for(i=1;i<=NF;i++)if($i=="group_id")g=i;next} NR>1&&g {print $g;exit}' "$1" | tr -d '\r"'
  }
  capture_diagnostics() {
    local label log out runroot latest_progress latest_session_log
    label=$1
    log=$2
    runroot=${3:-}
    out="$root/diagnostics/${label}_$(date -u +%Y%m%dT%H%M%SZ).txt"
    {
      echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      echo "label=$label"
      echo "===== nvidia-smi ====="; nvidia-smi || true
      echo "===== docker ====="; docker ps --format 'table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Image}}' || true
      echo "===== log ====="; tail -n 200 "$log" 2>/dev/null || true
      if [ -n "$runroot" ] && [ -d "$runroot" ]; then
        echo "===== Stage1 accepted and failed sessions ====="
        find "$runroot/status" -maxdepth 1 -type f \( -name '*.stage1.ok' -o -name '*.stage1.failed' \) -printf '%TY-%Tm-%Td %TH:%TM:%TS %f\n' 2>/dev/null | sort || true
        latest_progress=$(find "$runroot/status" -maxdepth 1 -name '*.progress.json' -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)
        echo "===== latest Stage1 progress: $latest_progress ====="
        if [ -n "$latest_progress" ]; then stat -c '%y %n' "$latest_progress"; cat "$latest_progress"; fi
        latest_session_log=$(find "$runroot/logs/stage1" "$runroot/logs/stage1_export" -maxdepth 1 -name '*.log' -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)
        echo "===== latest Stage1 session log: $latest_session_log ====="
        if [ -n "$latest_session_log" ]; then stat -c '%y %n' "$latest_session_log"; tail -n 120 "$latest_session_log"; fi
      fi
    } > "$out"
    echo "E9_DIAGNOSTIC=$out"
  }
  run_watched() {
    local label log watched_pid last_change last_size stalled size now watched_rc
    label=$1; log=$2; shift 2
    "$@" > "$log" 2>&1 &
    watched_pid=$!; last_change=$(date +%s); last_size=-1; stalled=0
    while kill -0 "$watched_pid" 2>/dev/null; do
      size=$(wc -c < "$log" 2>/dev/null || printf '0'); now=$(date +%s)
      if [ "$size" -ne "$last_size" ]; then last_size=$size; last_change=$now; fi
      if [ $((now-last_change)) -gt "$stall_seconds" ]; then
        echo "E9_STALL label=$label pid=$watched_pid"
        capture_diagnostics "$label-stall" "$log"
        kill -TERM "$watched_pid" 2>/dev/null || true
        stalled=1; break
      fi
      sleep 20
    done
    wait "$watched_pid"; watched_rc=$?
    if [ "$stalled" = 1 ]; then watched_rc=124; fi
    if [ "$watched_rc" -ne 0 ]; then capture_diagnostics "$label-failed" "$log"; return 1; fi
    return 0
  }
  wait_stage1_driver() {
    local label pid log run last_change last_size size now activity
    label=$1
    pid=$(cat /home/agni/LATEST_V4_STAGE1_OPT2_PID.txt 2>/dev/null)
    log=$(cat /home/agni/LATEST_V4_STAGE1_OPT2_LAUNCH_LOG.txt 2>/dev/null)
    run=$(cat /home/agni/LATEST_V4_STAGE1_OPT2_RUN.txt 2>/dev/null)
    last_change=$(date +%s)
    last_size=0
    while [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; do
      size=$(wc -c < "$log" 2>/dev/null || printf '0')
      now=$(date +%s)
      if [ "$size" -ne "$last_size" ]; then last_size=$size; last_change=$now; fi
      # The outer driver log is quiet while a Docker session writes per-slice
      # progress and its own log. Count those writes as real worker activity.
      activity=$(stage1_activity_epoch "$run")
      if [ -n "$activity" ] && [ "$activity" -gt "$last_change" ]; then
        last_change=$activity
      fi
      if [ $((now-last_change)) -gt "$stall_seconds" ]; then
        echo "E9_STAGE1_CONTROL_STALL label=$label pid=$pid last_activity_epoch=$last_change idle_seconds=$((now-last_change))"
        capture_diagnostics "$label-stall" "$log" "$run"
        kill -TERM "$pid" 2>/dev/null || true
        break
      fi
      printf '%s\n' "E9_STATUS=RUNNING_STAGE1_CONTROL label=$label pid=$pid" > "$root/STATE.txt"
      sleep 30
    done
    if [ -s "$run/STAGE1_OPT2_PRODUCTION_EQUIVALENT_COMPLETE.txt" ]; then
      printf '%s\n' "$run"
      return 0
    fi
    capture_diagnostics "$label-failed" "$log" "$run"
    return 1
  }
  launch_control_stage1() {
    local label expected only_gid control_stamp
    label=$1; expected=$2; only_gid=$3
    control_stamp=$(date -u +%Y%m%dT%H%M%SZ)
    RUN_STAMP="$control_stamp" VARIANT_NAME="$label" EXPECTED_SESSIONS="$expected" ONLY_GROUP_ID="$only_gid" IMAGE="$image" \
    DETAILED_CUDA_TIMING=0 RETAIN_GATHER_HOST_BUFFERS=1 CACHE_REFERENCE_COORDINATE_CHANNELS=1 USE_CUDA_GRAPH=0 \
    PREFETCH_INPUTS=0 PREPARE_CORE_SCHEDULE=0 PREFETCH_WORKERS=1 PREFETCH_DEPTH=1 ASYNC_OUTPUT_WRITES=0 \
    GROUPNORM_INPUT_LAYOUT=0 CONV_INPUT_LAYOUT=0 CHANNELS_LAST_WEIGHTS=0 KERNEL_FACTORY_INPUT_PACK=0 SCORE_ATOL=0 RESUME=0 \
      bash "$s1_ops/launch_v4_stage1_opt2_experiment.sh" >&2 || return
    wait_stage1_driver "$label"
  }
  run_disk_stage2() {
    label=$1; s1root=$2; outroot=$3
    s1c=$(host_to_container "$s1root"); outc=$(host_to_container "$outroot")
    bc=$(host_to_container "$bundle"); cc=$(host_to_container "$calibration"); pc=$(host_to_container "$profile")
    mapfile -t manifests < <(find "$s1root/stage1" -mindepth 2 -maxdepth 2 -name stage1_manifest.csv -type f | sort)
    for manifest in "${manifests[@]}"; do
      gid=$(gid_from_manifest "$manifest"); sid=$(basename "$(dirname "$manifest")")
      log="$root/control/logs/${label}_${sid}.log"; mkdir -p "$(dirname "$log")" "$outroot/$sid"
      run_watched "$label-stage2-$sid" "$log" timeout --signal=TERM --kill-after=120 "$session_timeout" docker run --rm \
        -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 \
        --mount "type=bind,source=$repo/v4,target=/workspace/v4,readonly" \
        --mount "type=bind,source=$s1_ops,target=/workspace/stage1opt,readonly" \
        --mount "type=bind,source=$s2_ops,target=/workspace/stage2,readonly" \
        --mount "type=bind,source=$e9_ops,target=/workspace/e9,readonly" \
        --mount "type=bind,source=$outputs,target=/outputs" \
        --workdir /workspace/v4 -e PYTHONPATH=/workspace/v4:/workspace/stage1opt:/workspace/stage2:/workspace/e9 \
        "$image" python /workspace/e9/run_e9_stage2_disk_control.py \
          --stage1_dir "$s1c/stage1/$sid" --output_dir "$outc/$sid" --session_filter "$gid" \
          --stage2_bundle "$bc" --calibration_json "$cc" --profile_json "$pc" \
          --timing_csv "$outc/timings/$sid.csv" --write_voxel_audit 1 --resume 1 || return 1
    done
    return 0
  }
  run_candidate_session() {
    label=$1; baseline_s1=$2; gid=$3; sid=$4; candidate_root=$5
    basec=$(host_to_container "$baseline_s1"); outc=$(host_to_container "$candidate_root/stage2/$sid")
    bc=$(host_to_container "$bundle"); cc=$(host_to_container "$calibration"); pc=$(host_to_container "$profile"); mc=$(host_to_container "$model")
    timingc=$(host_to_container "$candidate_root/timings/stage12/$sid.csv")
    stage2timingc=$(host_to_container "$candidate_root/timings/stage2/$sid.csv")
    qualityc=$(host_to_container "$candidate_root/status/$sid/quality")
    progressc=$(host_to_container "$candidate_root/status/$sid.progress.json")
    log="$candidate_root/logs/$sid.log"; mkdir -p "$(dirname "$log")" "$candidate_root/stage2/$sid" "$candidate_root/timings/stage12" "$candidate_root/timings/stage2" "$candidate_root/status/$sid"
    attempt=1
    while [ "$attempt" -le "$max_attempts" ]; do
      set +e
      timeout --signal=TERM --kill-after=120 "$session_timeout" docker run --rm --gpus all \
        -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 \
        --mount "type=bind,source=$repo/v4,target=/workspace/v4,readonly" \
        --mount "type=bind,source=$s1_ops,target=/workspace/stage1opt,readonly" \
        --mount "type=bind,source=$s2_ops,target=/workspace/stage2,readonly" \
        --mount "type=bind,source=$e9_ops,target=/workspace/e9,readonly" \
        --mount "type=bind,source=$outputs,target=/outputs" \
        --mount "type=bind,source=$input,target=/data/voxel_csv_combined,readonly" \
        --workdir /workspace/v4 -e PYTHONPATH=/workspace/v4:/workspace/stage1opt:/workspace/stage2:/workspace/e9 \
        "$image" python /workspace/e9/run_e9_stage12_inmemory.py \
          --e9-stage2-output-dir "$outc" --e9-stage2-bundle "$bc" --e9-stage2-profile "$pc" \
          --e9-baseline-stage1-dir "$basec" --e9-timing-csv "$timingc" --e9-stage2-timing-csv "$stage2timingc" --e9-quality-dir "$qualityc" \
          --e9-stage2-calibration-json "$cc" --e9-session-filter "$gid" \
          --input_dir /data/voxel_csv_combined --session_filter "$gid" --output_dir "/tmp/e9-stage1-$sid" \
          --model_path "$mc" --calibration_json "$cc" --timing_csv "/tmp/e9-stage1-$sid.csv" --progress_json "$progressc" \
          --batch_size 12 --amp bf16 --compile_model 0 --compile_mode default --channels_last 1 --pinned_d2h 0 \
          --detailed_cuda_timing 0 --retain_gather_host_buffers 1 --precompute_batch_gather_plans 0 \
          --cache_coordinate_channels 0 --cache_reference_coordinate_channels 1 --use_cuda_graph 0 \
          --prefetch_inputs 0 --prefetch_depth 1 --prefetch_workers 1 --prepare_core_schedule 0 --async_output_writes 0 \
          --groupnorm_input_layout 0 --conv_input_layout 0 --channels_last_weights 0 --kernel_factory_input_pack 0 \
          --prune_embedding_head 0 --warmup_iterations 0 --evaluate_all_cores 0 --gpu_coord_channels 1 --fixed_batch_shape 1 --resume 0 \
          > "$log" 2>&1 &
      worker=$!
      last_change=$(date +%s); last_size=0; stalled=0
      while kill -0 "$worker" 2>/dev/null; do
        now=$(date +%s); size=$(wc -c < "$log" 2>/dev/null || printf '0')
        progress_time=$(stat -c %Y "$candidate_root/status/$sid.progress.json" 2>/dev/null || printf '0')
        if [ "$size" -ne "$last_size" ] || [ "$progress_time" -ge "$last_change" ]; then
          last_size=$size; last_change=$now
        fi
        if [ $((now-last_change)) -gt "$stall_seconds" ]; then
          echo "E9_CANDIDATE_STALL sid=$sid attempt=$attempt worker=$worker"
          kill -TERM "$worker" 2>/dev/null || true
          stalled=1
          break
        fi
        sleep 20
      done
      wait "$worker"; rc=$?
      if [ "$stalled" = 1 ]; then rc=124; fi
      if [ "$rc" -eq 0 ] && [ -s "$candidate_root/stage2/$sid/E9_STAGE12_SESSION_COMPLETED.json" ]; then return 0; fi
      capture_diagnostics "$label-$sid-attempt$attempt" "$log"
      attempt=$((attempt+1))
    done
    return 1
  }
  compare_session() {
    local label control_dir candidate_dir sid report cc ic rc
    label=$1; control_dir=$2; candidate_dir=$3; sid=$4
    report="$root/comparisons/${label}_${sid}.json"
    cc=$(host_to_container "$control_dir"); ic=$(host_to_container "$candidate_dir"); rc=$(host_to_container "$report")
    docker run --rm --mount "type=bind,source=$repo/v4,target=/workspace/v4,readonly" --mount "type=bind,source=$e9_ops,target=/workspace/e9,readonly" --mount "type=bind,source=$outputs,target=/outputs" \
      --workdir /workspace/v4 -e PYTHONPATH=/workspace/v4:/workspace/e9 "$image" \
      python /workspace/e9/compare_e9_stage2_outputs.py --control "$cc" --candidate "$ic" --report "$rc"
  }
  run_candidate_all() {
    label=$1; control_label=$2; control_s1=$3; candidate_root=$4; expected=$5
    mapfile -t manifests < <(find "$control_s1/stage1" -mindepth 2 -maxdepth 2 -name stage1_manifest.csv -type f | sort)
    if [ "${#manifests[@]}" -ne "$expected" ]; then echo "E9_STATUS=STOP_SESSION_COUNT control=${#manifests[@]} expected=$expected"; return 1; fi
    count=0
    for manifest in "${manifests[@]}"; do
      gid=$(gid_from_manifest "$manifest"); sid=$(basename "$(dirname "$manifest")")
      baseline_s1="$baseline/stage1/$sid"
      run_candidate_session "$label" "$baseline_s1" "$gid" "$sid" "$candidate_root" || return 1
      compare_session "$label" "$root/control/$control_label/stage2/$sid" "$candidate_root/stage2/$sid" "$sid" || return 1
      count=$((count+1)); printf '%s\n' "E9_STATUS=RUNNING_CANDIDATE label=$label sessions=$count/$expected" > "$root/STATE.txt"
    done
    return 0
  }
  # Two repeated representative checks catch nondeterminism before all-session work.
  for repeat in 1 2; do
    c_label="e9_control_r$repeat"; i_label="e9_inmemory_r$repeat"
    control_s1=$(launch_control_stage1 "$c_label" 1 "$group") || { printf '%s\n' "E9_STATUS=REJECTED_${c_label}" > "$root/STATE.txt"; return; }
    run_disk_stage2 "$c_label" "$control_s1" "$root/control/$c_label/stage2" || { printf '%s\n' "E9_STATUS=REJECTED_STAGE2_${c_label}" > "$root/STATE.txt"; return; }
    run_candidate_all "$i_label" "$c_label" "$control_s1" "$root/candidate/$i_label" 1 || { printf '%s\n' "E9_STATUS=REJECTED_${i_label}" > "$root/STATE.txt"; return; }
  done
  control_s1=$(launch_control_stage1 e9_control_full30 30 "") || { printf '%s\n' "E9_STATUS=REJECTED_CONTROL_FULL30" > "$root/STATE.txt"; return; }
  run_disk_stage2 e9_control_full30 "$control_s1" "$root/control/e9_control_full30/stage2" || { printf '%s\n' "E9_STATUS=REJECTED_STAGE2_CONTROL_FULL30" > "$root/STATE.txt"; return; }
  run_candidate_all e9_inmemory_full30 e9_control_full30 "$control_s1" "$root/candidate/e9_inmemory_full30" 30 || { printf '%s\n' "E9_STATUS=REJECTED_INMEMORY_FULL30" > "$root/STATE.txt"; return; }
  controlc=$(host_to_container "$control_s1/timings/stage1"); stage2c=$(host_to_container "$root/control/e9_control_full30/stage2/timings"); candidatec=$(host_to_container "$root/candidate/e9_inmemory_full30/timings/stage12"); comparisonsc=$(host_to_container "$root/comparisons"); jsonc=$(host_to_container "$root/summaries/E9_RESULT.json"); mdc=$(host_to_container "$root/summaries/E9_RESULT.md")
  docker run --rm --mount "type=bind,source=$repo/v4,target=/workspace/v4,readonly" --mount "type=bind,source=$e9_ops,target=/workspace/e9,readonly" --mount "type=bind,source=$outputs,target=/outputs" --workdir /workspace/v4 -e PYTHONPATH=/workspace/v4:/workspace/e9 "$image" \
    python /workspace/e9/summarize_e9_stage12.py --control-stage1-timings "$controlc" --control-stage2-timings "$stage2c" --candidate-timings "$candidatec" --comparison-dir "$comparisonsc" --json "$jsonc" --markdown "$mdc"
  decision=$(sed -n 's/^[[:space:]]*"decision": "\([^"]*\)".*/\1/p' "$root/summaries/E9_RESULT.json")
  printf '%s\n' "E9_STATUS=$decision" > "$root/STATE.txt"
  cp "$control_s1/STAGE1_TIMING_SESSION_AVERAGES.txt" "$root/summaries/CONTROL_STAGE1_TIMING_SESSION_AVERAGES.txt" 2>/dev/null || true
  mkdir -p "$root/summaries/control_stage1_raw_timings"
  cp "$control_s1/timings/stage1/"*.csv "$root/summaries/control_stage1_raw_timings/" || {
    echo "E9_STATUS=STOP_CONTROL_RAW_TIMINGS_MISSING" > "$root/STATE.txt"
    return
  }
  archive="/home/agni/v4_stage12_e9_${stamp}.tar.gz"
  if ! tar -czf "$archive.partial" -C "$root" STATE.txt E9_HARNESS.log summaries comparisons diagnostics candidate/e9_inmemory_full30/timings candidate/e9_inmemory_full30/status control/e9_control_full30/stage2/timings; then
    echo "E9_STATUS=STOP_RESULT_PACKAGE_FAILED" > "$root/STATE.txt"
    echo "E9_RESULT_PACKAGE_FAILED partial=$archive.partial"
    return
  fi
  mv "$archive.partial" "$archive"
  sha256sum "$archive" > "$archive.sha256"
  printf '%s\n' "$archive" > /home/agni/LATEST_V4_STAGE12_E9_RESULT_ARCHIVE.txt
  echo "E9_RESULT_ARCHIVE=$archive"
  echo "E9_DECISION=$decision"
}

main "$@"
