#!/usr/bin/env bash
# Build a compact review package after success OR failure.
# Host shell packages files; any Python diagnostics execute inside Docker.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/v4_nebius_common.sh"
v4_print_context

TEST_ROOT_HOST=$(v4_latest_test_root_host 2>/dev/null || true)
REPLAY_HOST=''
if [[ -n "$TEST_ROOT_HOST" ]]; then
  if [[ -f "$TEST_ROOT_HOST/replay_full/realtime_slice_timing.csv" ]]; then REPLAY_HOST="$TEST_ROOT_HOST/replay_full"
  elif [[ -f "$TEST_ROOT_HOST/replay_quick/realtime_slice_timing.csv" ]]; then REPLAY_HOST="$TEST_ROOT_HOST/replay_quick"
  fi
fi
if [[ -z "$REPLAY_HOST" && -f "$V4_RUN_ROOT_HOST/realtime_slice_timing.csv" ]]; then REPLAY_HOST="$V4_RUN_ROOT_HOST"; fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
DIAG_OUT="$V4_RUN_ROOT/review_realtime_diagnostics.json"
if [[ -n "$REPLAY_HOST" ]]; then
  REPLAY=$(v4_to_container_output "$REPLAY_HOST")
  # Verification/summarization are best-effort for a failure package; collection must still run.
  v4_docker_run "set +e; \
    python verify_v4_realtime_replay.py --replay_dir '$REPLAY' --max_sequence_gap 9 --slice_length_ft 50 --max_span_length_ft 450 > '$REPLAY/review_verify.log' 2>&1; VERIFY_RC=\$?; \
    python summarize_v4_realtime_timing.py --timing_csv '$REPLAY/realtime_slice_timing.csv' --output_json '$REPLAY/realtime_timing_summary.json' --output_txt '$REPLAY/realtime_timing_summary.txt' > '$REPLAY/review_timing.log' 2>&1; TIMING_RC=\$?; \
    set -e; python collect_v4_realtime_diagnostics.py --run_root '$V4_RUN_ROOT' --replay_dir '$REPLAY' --diagnostics_dir '$V4_DIAG_ROOT' --output '$DIAG_OUT'; \
    printf 'verify_rc=%s\\ntiming_rc=%s\\n' \"\$VERIFY_RC\" \"\$TIMING_RC\" > '$REPLAY/review_packaging_status.txt'"
else
  v4_docker_run "python collect_v4_realtime_diagnostics.py --run_root '$V4_RUN_ROOT' --diagnostics_dir '$V4_DIAG_ROOT' --output '$DIAG_OUT'"
fi

STAGE="$V4_REVIEW_ROOT_HOST/.stage_${V4_SESSION_SAFE}_${V4_DEPLOY_SHORT}_$$"
ARCHIVE="$V4_REVIEW_ROOT_HOST/v4_review_${V4_SESSION_SAFE}_${V4_DEPLOY_SHORT}_${STAMP}.tar.gz"
rm -rf "$STAGE"; mkdir -p "$STAGE/context" "$STAGE/diagnostics" "$STAGE/replay" "$STAGE/test" "$STAGE/preflight"
trap 'rm -rf "$STAGE"' EXIT
cp -f "$V4_RUN_ROOT_HOST/run_context.env" "$STAGE/context/" 2>/dev/null || true
cp -f "$V4_SESSION_INVENTORY_HOST" "$STAGE/context/" 2>/dev/null || true
cp -f "$V4_SESSION_ROOT_HOST/LATEST_RUN_ROOT.txt" "$STAGE/context/" 2>/dev/null || true
cp -a "$V4_DIAG_ROOT_HOST/." "$STAGE/diagnostics/" 2>/dev/null || true
cp -f "$V4_RUN_ROOT_HOST/review_realtime_diagnostics.json" "$STAGE/diagnostics/" 2>/dev/null || true

PREFLIGHT_HOST=$(cat "$V4_RUN_ROOT_HOST/LATEST_PREFLIGHT_ROOT.txt" 2>/dev/null || true)
if [[ -n "$PREFLIGHT_HOST" && -d "$PREFLIGHT_HOST" ]]; then
  cp -f "$PREFLIGHT_HOST"/preflight.log "$PREFLIGHT_HOST"/PREFLIGHT_OK.txt "$PREFLIGHT_HOST"/PREFLIGHT_FAILED.txt "$STAGE/preflight/" 2>/dev/null || true
fi

if [[ -n "$REPLAY_HOST" && -d "$REPLAY_HOST" ]]; then
  for f in COMPLETED.json FAILED.json REALTIME_REPLAY_VERIFICATION.json realtime_slice_timing.csv realtime_timing_summary.json realtime_timing_summary.txt review_verify.log review_timing.log review_packaging_status.txt review_realtime_diagnostics.json inference_manifest.csv stage1_manifest.csv stage2_manifest.csv production_runner.log benchmark.log; do
    [[ -f "$REPLAY_HOST/$f" ]] && cp -f "$REPLAY_HOST/$f" "$STAGE/replay/$f"
  done
  [[ -d "$REPLAY_HOST/errors" ]] && cp -a "$REPLAY_HOST/errors" "$STAGE/replay/errors"
  LATEST_JSON=$(find "$REPLAY_HOST/stage3_incremental" -type f -name LATEST.json -print 2>/dev/null | sort | tail -1 || true)
  if [[ -n "$LATEST_JSON" ]]; then
    cp -f "$LATEST_JSON" "$STAGE/replay/LATEST.json"
    SNAP_ROOT=$(dirname "$LATEST_JSON")
    SNAP=$(find "$SNAP_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'slice_*' -print 2>/dev/null | sort | tail -1 || true)
    [[ -n "$SNAP" ]] && cp -a "$SNAP" "$STAGE/replay/latest_stage3_snapshot"
  fi
fi

if [[ -n "$TEST_ROOT_HOST" && -d "$TEST_ROOT_HOST" ]]; then
  for f in acceptance.log recovery.log RECOVERY_ACCEPTANCE_OK.txt recovery_context.txt quick_reused_REALTIME_REPLAY_VERIFICATION.json quick_reused_timing_summary.json quick_reused_timing_summary.txt FAILED.txt runtime_variant_equivalence.json v4_runtime_mode.env v4_batch_size.env; do
    [[ -f "$TEST_ROOT_HOST/$f" ]] && cp -f "$TEST_ROOT_HOST/$f" "$STAGE/test/$f"
  done
  [[ -d "$TEST_ROOT_HOST/batch_sweep" ]] && cp -a "$TEST_ROOT_HOST/batch_sweep" "$STAGE/test/batch_sweep"
  [[ -d "$TEST_ROOT_HOST/prior_evidence" ]] && cp -a "$TEST_ROOT_HOST/prior_evidence" "$STAGE/test/prior_evidence"
  [[ -f "$TEST_ROOT_HOST/independent_stage1_relocated/STAGE1_COMPLETED.json" ]] && cp -f "$TEST_ROOT_HOST/independent_stage1_relocated/STAGE1_COMPLETED.json" "$STAGE/test/STAGE1_COMPLETED.json"
  [[ -f "$TEST_ROOT_HOST/independent_stage2_relocated/STAGE2_COMPLETED.json" ]] && cp -f "$TEST_ROOT_HOST/independent_stage2_relocated/STAGE2_COMPLETED.json" "$STAGE/test/STAGE2_COMPLETED.json"
  [[ -f "$TEST_ROOT_HOST/independent_stage3/STAGE3_COMPLETED.json" ]] && cp -f "$TEST_ROOT_HOST/independent_stage3/STAGE3_COMPLETED.json" "$STAGE/test/STAGE3_COMPLETED.json"
  [[ -f "$TEST_ROOT_HOST/independent_stage3_repeat/STAGE3_COMPLETED.json" ]] && cp -f "$TEST_ROOT_HOST/independent_stage3_repeat/STAGE3_COMPLETED.json" "$STAGE/test/STAGE3_REPEAT_COMPLETED.json"
fi

cat > "$STAGE/REVIEW_PACKAGE_INFO.txt" <<INFO
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
session=$V4_SESSION_FILTER
deployment_fingerprint=$V4_DEPLOY_FINGERPRINT
code_fingerprint=$V4_CODE_FINGERPRINT
runtime_mode=$V4_RUNTIME_MODE
batch_size=$V4_BATCH_SIZE
replay_host=${REPLAY_HOST:-none}
test_root_host=${TEST_ROOT_HOST:-none}
preflight_host=${PREFLIGHT_HOST:-none}
acceptance_fingerprint_match=$V4_ACCEPTANCE_FINGERPRINT_MATCH
max_sequence_gap=9
slice_length_ft=50
max_span_length_ft=450
max_observed_slice_centers=10
INFO

tar -C "$STAGE" -czf "$ARCHIVE" .
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
printf '%s\n' "$ARCHIVE" > "$V4_REVIEW_ROOT_HOST/LATEST_REVIEW_ARCHIVE.txt"
printf '%s\n' "$ARCHIVE.sha256" > "$V4_REVIEW_ROOT_HOST/LATEST_REVIEW_SHA256.txt"
echo "Review archive: $ARCHIVE"
echo "SHA256 file:   $ARCHIVE.sha256"
echo "Latest pointer: $V4_REVIEW_ROOT_HOST/LATEST_REVIEW_ARCHIVE.txt"
ls -lh "$ARCHIVE" "$ARCHIVE.sha256"
