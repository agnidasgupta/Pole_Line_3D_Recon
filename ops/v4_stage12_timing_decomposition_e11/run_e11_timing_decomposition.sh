#!/usr/bin/env bash
# E11: two repeated representative-session timing decompositions.
# This harness never starts a full-30 run and never alters production V4.

main() {
  mode=${1:---run}
  repo=${REPO:-/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10}
  outputs=${HOST_OUTPUTS:-/workspace/voxel_poleline/outputs}
  input=${HOST_INPUT:-/data/voxel_csv_combined}
  image=${IMAGE:-va-v4-realtime:torch241-cu121}
  baseline=${BASELINE_RUN_HOST:-$outputs/poleline_voxel_run_session_groups/v4_production/full_dataset_runs/d9977c39c443f5fa14f8/20260825T203403Z}
  model=${MODEL_HOST:-$outputs/poleline_voxel_run_session_groups/precision_v4/train/precision_best.pt}
  calibration=${CALIBRATION_HOST:-$outputs/poleline_voxel_run_session_groups/precision_v4/full_val/calibration.json}
  bundle=${STAGE2_BUNDLE:-$outputs/poleline_voxel_run_session_groups/v4_realtime/stage2_refiner/local_refiner_bundle.joblib}
  profile=${STAGE2_PROFILE:-$outputs/poleline_voxel_run_session_groups/v4_stage23_quality/full_run_v2_20260904T194258Z/selection/selected_electrical_profile.json}
  group=${E11_GROUP_ID:-NYSEGDistVegMgnt_AUBURN_-_SPRUCE_HAVEN_FARMS_TAP_520-3p_2029/session2}
  s1_ops="$repo/ops/v4_stage1_inference_opt2"
  s2_ops="$repo/ops/v4_stage2_stage1_electrical_v10"
  e9_ops="$repo/ops/v4_stage12_inmemory_handoff_e9"
  e10_ops="$repo/ops/v4_stage12_persistent_stage2_e10"
  e11_ops="$repo/ops/v4_stage12_timing_decomposition_e11"

  if [ "$mode" = --monitor ]; then
    root=${2:-$(cat /home/agni/LATEST_V4_STAGE12_E11_HARNESS.txt 2>/dev/null)}
    echo "E11_HARNESS_ROOT=$root"
    cat "$root/STATE.txt" 2>/dev/null || true
    echo '===== RECENT HARNESS LOG ====='
    tail -n 120 "$root/E11_HARNESS.log" 2>/dev/null || true
    return
  fi
  if [ "$mode" = --self-test ]; then
    bash -n "$e11_ops/run_e11_timing_decomposition.sh" || return 1
    docker run --rm \
      --mount "type=bind,source=$e9_ops,target=/workspace/e9,readonly" \
      --mount "type=bind,source=$e10_ops,target=/workspace/e10,readonly" \
      --mount "type=bind,source=$e11_ops,target=/workspace/e11,readonly" \
      --workdir /workspace -e PYTHONPATH=/workspace/e9:/workspace/e10:/workspace/e11 \
      "$image" python -m py_compile /workspace/e11/run_e11_stage12_profile.py /workspace/e11/run_e11_stage2_disk_profile.py /workspace/e11/summarize_e11_timing_decomposition.py || return 1
    echo E11_SELF_TEST_OK
    return
  fi
  for path in "$repo/.git" "$s1_ops/run_v4_stage1_opt2.py" "$s2_ops/run_v4_stage2_stage1_electrical_tracks.py" "$e11_ops/run_e11_stage12_profile.py" "$baseline/PHASE1_STAGE1_OK.txt" "$input" "$model" "$calibration" "$bundle" "$profile"; do
    if [ ! -e "$path" ]; then echo "E11_STATUS=STOP_MISSING_REQUIRED_PATH path=$path"; return 1; fi
  done
  for asset in \
    "$model:1b8b20c0bb2b52a1617555ed72c34311ba3839effd674bb2cac5273040d909ee" \
    "$calibration:dea4829143f33d1f674176185ecd59df620c50a70488a83b0a2d6e17b81784e1" \
    "$bundle:c451d501c3a3ccf7598ce8254d4807483afe3f41e4500be8a9abcf705843e72d" \
    "$profile:de79c637e9d70ff0d39c2765b8b6514f08a8547374079cd1f9f803cb9879ca1d"; do
    path=${asset%:*}; expected=${asset##*:}; actual=$(sha256sum "$path" | awk '{print $1}')
    [ "$actual" = "$expected" ] || { echo "E11_STATUS=STOP_ASSET_SHA_MISMATCH path=$path"; return 1; }
  done
  host_to_container() { case "$1" in "$outputs"/*) printf '/outputs/%s' "${1#"$outputs"/}" ;; *) printf '%s' "$1" ;; esac; }
  capture() { mkdir -p "$root/diagnostics"; { echo "label=$1"; echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; tail -n 160 "$2" 2>/dev/null || true; } > "$root/diagnostics/$1-$(date -u +%Y%m%dT%H%M%SZ).txt"; }
  wait_stage1() {
    label=$1; pid=$(cat /home/agni/LATEST_V4_STAGE1_OPT2_PID.txt 2>/dev/null); log=$(cat /home/agni/LATEST_V4_STAGE1_OPT2_LAUNCH_LOG.txt 2>/dev/null); run=$(cat /home/agni/LATEST_V4_STAGE1_OPT2_RUN.txt 2>/dev/null)
    while [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; do printf '%s\n' "E11_STATUS=RUNNING_STAGE1 label=$label pid=$pid" > "$root/STATE.txt"; sleep 30; done
    if [ -s "$run/STAGE1_OPT2_PRODUCTION_EQUIVALENT_COMPLETE.txt" ]; then printf '%s\n' "$run"; return 0; fi
    capture "$label-stage1-failed" "$log"; return 1
  }
  launch_stage1() {
    label=$1; stamp=$(date -u +%Y%m%dT%H%M%SZ)
    RUN_STAMP="$stamp" VARIANT_NAME="$label" EXPECTED_SESSIONS=1 ONLY_GROUP_ID="$group" IMAGE="$image" \
    DETAILED_CUDA_TIMING=0 RETAIN_GATHER_HOST_BUFFERS=1 CACHE_REFERENCE_COORDINATE_CHANNELS=1 USE_CUDA_GRAPH=0 \
    PREFETCH_INPUTS=0 PREPARE_CORE_SCHEDULE=0 PREFETCH_WORKERS=1 PREFETCH_DEPTH=1 ASYNC_OUTPUT_WRITES=0 \
    GROUPNORM_INPUT_LAYOUT=0 CONV_INPUT_LAYOUT=0 CHANNELS_LAST_WEIGHTS=0 KERNEL_FACTORY_INPUT_PACK=0 SCORE_ATOL=0 RESUME=0 \
      bash "$s1_ops/launch_v4_stage1_opt2_experiment.sh" >&2 || return 1
    wait_stage1 "$label"
  }
  run_disk() {
    label=$1; s1_session=$2; out=$3; timing=$4; log="$root/logs/${label}_disk.log"
    mkdir -p "$(dirname "$log")" "$out" "$(dirname "$timing")"
    s1c=$(host_to_container "$s1_session"); outc=$(host_to_container "$out"); tc=$(host_to_container "$timing"); bc=$(host_to_container "$bundle"); cc=$(host_to_container "$calibration"); pc=$(host_to_container "$profile")
    docker run --rm \
      -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 \
      --mount "type=bind,source=$repo/v4,target=/workspace/v4,readonly" --mount "type=bind,source=$s2_ops,target=/workspace/stage2,readonly" \
      --mount "type=bind,source=$e9_ops,target=/workspace/e9,readonly" --mount "type=bind,source=$e11_ops,target=/workspace/e11,readonly" \
      --mount "type=bind,source=$outputs,target=/outputs" --workdir /workspace/v4 -e PYTHONPATH=/workspace/v4:/workspace/stage2:/workspace/e9:/workspace/e11 \
      "$image" python /workspace/e11/run_e11_stage2_disk_profile.py --stage1_dir "$s1c" --output_dir "$outc" --session_filter "$group" --stage2_bundle "$bc" --calibration_json "$cc" --profile_json "$pc" --timing_csv "$tc" --write_voxel_audit 1 --resume 0 > "$log" 2>&1 || { capture "$label-disk-failed" "$log"; return 1; }
  }
  run_candidate() {
    label=$1; baseline_s1=$2; sid=$3; out=$4; timing=$5; s2timing=$6; quality=$7; log="$root/logs/${label}_candidate.log"
    mkdir -p "$(dirname "$log")" "$out" "$(dirname "$timing")" "$(dirname "$s2timing")" "$quality"
    basec=$(host_to_container "$baseline_s1"); outc=$(host_to_container "$out"); tc=$(host_to_container "$timing"); s2tc=$(host_to_container "$s2timing"); qc=$(host_to_container "$quality"); bc=$(host_to_container "$bundle"); cc=$(host_to_container "$calibration"); pc=$(host_to_container "$profile"); mc=$(host_to_container "$model")
    docker run --rm --gpus all \
      -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 \
      --mount "type=bind,source=$repo/v4,target=/workspace/v4,readonly" --mount "type=bind,source=$s1_ops,target=/workspace/stage1opt,readonly" --mount "type=bind,source=$s2_ops,target=/workspace/stage2,readonly" \
      --mount "type=bind,source=$e9_ops,target=/workspace/e9,readonly" --mount "type=bind,source=$e10_ops,target=/workspace/e10,readonly" --mount "type=bind,source=$e11_ops,target=/workspace/e11,readonly" \
      --mount "type=bind,source=$outputs,target=/outputs" --mount "type=bind,source=$input,target=/data/voxel_csv_combined,readonly" \
      --workdir /workspace/v4 -e PYTHONPATH=/workspace/v4:/workspace/stage1opt:/workspace/stage2:/workspace/e9:/workspace/e10:/workspace/e11 \
      "$image" python /workspace/e11/run_e11_stage12_profile.py \
        --e9-stage2-output-dir "$outc" --e9-stage2-bundle "$bc" --e9-stage2-profile "$pc" --e9-baseline-stage1-dir "$basec" --e9-timing-csv "$tc" --e9-stage2-timing-csv "$s2tc" --e9-quality-dir "$qc" --e9-stage2-calibration-json "$cc" --e9-session-filter "$group" \
        --input_dir /data/voxel_csv_combined --session_filter "$group" --output_dir "/tmp/e11-stage1-$sid" --model_path "$mc" --calibration_json "$cc" --timing_csv "/tmp/e11-stage1-$sid.csv" --progress_json "/tmp/e11-progress-$sid.json" \
        --batch_size 12 --amp bf16 --compile_model 0 --compile_mode default --channels_last 1 --pinned_d2h 0 --detailed_cuda_timing 0 --retain_gather_host_buffers 1 --precompute_batch_gather_plans 0 \
        --cache_coordinate_channels 0 --cache_reference_coordinate_channels 1 --use_cuda_graph 0 --prefetch_inputs 0 --prefetch_depth 1 --prefetch_workers 1 --prepare_core_schedule 0 --async_output_writes 0 \
        --groupnorm_input_layout 0 --conv_input_layout 0 --channels_last_weights 0 --kernel_factory_input_pack 0 --prune_embedding_head 0 --warmup_iterations 0 --evaluate_all_cores 0 --gpu_coord_channels 1 --fixed_batch_shape 1 --resume 0 > "$log" 2>&1 || { capture "$label-candidate-failed" "$log"; return 1; }
  }
  compare() {
    label=$1; control=$2; candidate=$3; report="$root/comparisons/$label.json"; mkdir -p "$(dirname "$report")"
    cc=$(host_to_container "$control"); ic=$(host_to_container "$candidate"); rc=$(host_to_container "$report")
    docker run --rm --mount "type=bind,source=$repo/v4,target=/workspace/v4,readonly" --mount "type=bind,source=$e9_ops,target=/workspace/e9,readonly" --mount "type=bind,source=$outputs,target=/outputs" --workdir /workspace/v4 -e PYTHONPATH=/workspace/v4:/workspace/e9 "$image" python /workspace/e9/compare_e9_stage2_outputs.py --control "$cc" --candidate "$ic" --report "$rc"
  }

  stamp=${E11_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}; root="$outputs/v4_stage12_timing_decomposition_e11/$stamp"
  mkdir -p "$root"/{control,candidate,timings/control_stage1,timings/control_stage2,timings/candidate,comparisons,logs,diagnostics,summaries}
  printf '%s\n' "$root" > /home/agni/LATEST_V4_STAGE12_E11_HARNESS.txt
  exec > >(tee -a "$root/E11_HARNESS.log") 2>&1
  echo "E11_HARNESS_ROOT=$root"; echo 'E11_CONTRACT=exact_stage1_payload;all_stage2_outputs;serial_refiner;no_candidate_stage1_artifacts;two_representative_repeats'
  for repeat in 1 2; do
    label="e11_r$repeat"; printf '%s\n' "E11_STATUS=RUNNING repeat=$repeat/2" > "$root/STATE.txt"
    control_run=$(launch_stage1 "${label}_control") || { printf '%s\n' "E11_STATUS=REJECTED_STAGE1_CONTROL" > "$root/STATE.txt"; return; }
    manifest=$(find "$control_run/stage1" -mindepth 2 -maxdepth 2 -name stage1_manifest.csv -type f | head -n 1)
    [ -n "$manifest" ] || { printf '%s\n' "E11_STATUS=STOP_CONTROL_MANIFEST_MISSING" > "$root/STATE.txt"; return; }
    sid=$(basename "$(dirname "$manifest")"); s1_session=$(dirname "$manifest"); baseline_s1="$baseline/stage1/$sid"
    [ -d "$baseline_s1" ] || { printf '%s\n' "E11_STATUS=STOP_BASELINE_SESSION_MISSING sid=$sid" > "$root/STATE.txt"; return; }
    cp "$control_run/timings/stage1/"*.csv "$root/timings/control_stage1/$label.csv" || { printf '%s\n' 'E11_STATUS=STOP_CONTROL_STAGE1_TIMING_MISSING' > "$root/STATE.txt"; return; }
    disk_root="$root/control/$label/stage2/$sid"; disk_time="$root/timings/control_stage2/$label.csv"
    run_disk "$label" "$s1_session" "$disk_root" "$disk_time" || { printf '%s\n' 'E11_STATUS=REJECTED_DISK_CONTROL' > "$root/STATE.txt"; return; }
    candidate_root="$root/candidate/$label/stage2/$sid"; candidate_time="$root/timings/candidate/$label.csv"
    run_candidate "$label" "$baseline_s1" "$sid" "$candidate_root" "$candidate_time" "$root/candidate/$label/stage2_timing.csv" "$root/candidate/$label/quality" || { printf '%s\n' 'E11_STATUS=REJECTED_CANDIDATE' > "$root/STATE.txt"; return; }
    compare "$label" "$disk_root" "$candidate_root" || { printf '%s\n' 'E11_STATUS=REJECTED_STAGE2_OUTPUTS' > "$root/STATE.txt"; return; }
  done
  c1=$(host_to_container "$root/timings/control_stage1"); c2=$(host_to_container "$root/timings/control_stage2"); cand=$(host_to_container "$root/timings/candidate"); comp=$(host_to_container "$root/comparisons"); outj=$(host_to_container "$root/summaries/E11_RESULT.json"); outm=$(host_to_container "$root/summaries/E11_RESULT.md")
  docker run --rm --mount "type=bind,source=$e11_ops,target=/workspace/e11,readonly" --mount "type=bind,source=$outputs,target=/outputs" --workdir /workspace/e11 -e PYTHONPATH=/workspace/e11 "$image" python /workspace/e11/summarize_e11_timing_decomposition.py --control-stage1 "$c1" --control "$c2" --candidate "$cand" --comparisons "$comp" --json "$outj" --markdown "$outm" || { printf '%s\n' 'E11_STATUS=STOP_SUMMARY_FAILED' > "$root/STATE.txt"; return; }
  decision=$(sed -n 's/^[[:space:]]*"decision": "\([^"]*\)".*/\1/p' "$root/summaries/E11_RESULT.json")
  printf '%s\n' "E11_STATUS=$decision" > "$root/STATE.txt"
  archive="/home/agni/v4_stage12_e11_${stamp}.tar.gz"; tar -czf "$archive" -C "$root" STATE.txt E11_HARNESS.log summaries comparisons diagnostics timings logs
  sha256sum "$archive" > "$archive.sha256"; printf '%s\n' "$archive" > /home/agni/LATEST_V4_STAGE12_E11_RESULT_ARCHIVE.txt
  echo "E11_RESULT_ARCHIVE=$archive"; echo "E11_DECISION=$decision"
}

main "$@"
