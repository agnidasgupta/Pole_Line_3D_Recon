#!/usr/bin/env bash
# Detached full-session benchmark using the already gated runtime and auto-selected session.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/v4_nebius_common.sh"
v4_print_context
v4_require_current_gate
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
BENCH_HOST="$V4_RUN_ROOT_HOST/benchmarks/$STAMP"
mkdir -p "$BENCH_HOST"
BENCH=$(v4_to_container_output "$BENCH_HOST")
NAME=${NAME:-v4-bench-$(printf '%s' "$V4_SESSION_SAFE" | tr -cd 'A-Za-z0-9_.-' | cut -c1-40)-${V4_DEPLOY_SHORT:0:8}}
CMD="python run_v4_realtime_session.py --input_dir /data/voxel_csv_combined --session_filter '$V4_SESSION_FILTER' --output_dir '$BENCH' --model_path '$V4_MODEL' --calibration_json '$V4_CAL' --stage2_bundle '$V4_STAGE2' --batch_size '$V4_BATCH_SIZE' --amp '${AMP:-bf16}' --evaluate_all_cores '$V4_EVALUATE_ALL_CORES' --gpu_coord_channels '$V4_GPU_COORD_CHANNELS' --fixed_batch_shape '$V4_FIXED_BATCH_SHAPE' --max_sequence_gap 9 --slice_length_ft 50 --max_span_length_ft 450 --stage3_every_slice 1 --stage3_execution inprocess --stage3_inmemory_cache 1 --resume '${RESUME:-0}' --max_slices '${MAX_SLICES:-0}'; python verify_v4_realtime_replay.py --replay_dir '$BENCH' --max_sequence_gap 9 --slice_length_ft 50 --max_span_length_ft 450; python summarize_v4_realtime_timing.py --timing_csv '$BENCH/realtime_slice_timing.csv' --output_json '$BENCH/realtime_timing_summary.json' --output_txt '$BENCH/realtime_timing_summary.txt'"
v4_detached_run "$NAME" "$BENCH/benchmark.log" "$CMD"
printf '%s\n' "$BENCH_HOST" > "$V4_RUN_ROOT_HOST/LATEST_BENCHMARK_ROOT.txt"
echo "Benchmark root: $BENCH_HOST"
