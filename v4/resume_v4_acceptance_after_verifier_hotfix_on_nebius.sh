#!/usr/bin/env bash
# Resume the 2026-08-25 replay-verifier-only acceptance failure without repeating
# the H100 runtime gate, but only after proving runtime/model identity.
# Host side uses shell only. Every Python command runs inside Docker.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/v4_nebius_common.sh"
v4_print_context

# Find the newest *original acceptance* failure at the replay verifier. Do not
# depend on LATEST_TEST_ROOT because a later recovery attempt may have updated it.
OLD_TEST_ROOT=''
while IFS= read -r fail; do
  cand=$(dirname "$fail")
  if grep -Fq 'verify_v4_realtime_replay.py' "$fail" \
    && [[ -s "$cand/runtime_variant_equivalence.json" ]] \
    && [[ -s "$cand/v4_runtime_mode.env" ]] \
    && [[ -s "$cand/v4_batch_size.env" ]] \
    && [[ -s "$cand/replay_quick/COMPLETED.json" ]]; then
    OLD_TEST_ROOT=$cand
    break
  fi
done < <(find "$V4_SESSION_ROOT_HOST/runs" -type f -path '*/tests/*/FAILED.txt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | cut -d' ' -f2-)
[[ -n "$OLD_TEST_ROOT" && -d "$OLD_TEST_ROOT" ]] || {
  echo "ERROR: no prior replay-verifier-only acceptance failure with reusable gate evidence was found for $V4_SESSION_FILTER" >&2
  exit 2
}
for p in \
  "$OLD_TEST_ROOT/runtime_variant_equivalence.json" \
  "$OLD_TEST_ROOT/v4_runtime_mode.env" \
  "$OLD_TEST_ROOT/v4_batch_size.env" \
  "$OLD_TEST_ROOT/independent_stage1_relocated/STAGE1_COMPLETED.json" \
  "$OLD_TEST_ROOT/independent_stage2_relocated/STAGE2_COMPLETED.json" \
  "$OLD_TEST_ROOT/independent_stage3/STAGE3_COMPLETED.json" \
  "$OLD_TEST_ROOT/independent_stage3_repeat/STAGE3_COMPLETED.json" \
  "$OLD_TEST_ROOT/replay_quick/COMPLETED.json"; do
  [[ -s "$p" ]] || { echo "ERROR: required prior evidence missing: $p" >&2; exit 3; }
done

OLD_RUN_ROOT=$(dirname "$(dirname "$OLD_TEST_ROOT")")
OLD_CONTEXT="$OLD_RUN_ROOT/run_context.env"
[[ -s "$OLD_CONTEXT" ]] || { echo "ERROR: prior run_context.env missing: $OLD_CONTEXT" >&2; exit 3; }
old_value() { bash -c 'set -euo pipefail; source "$1"; eval "printf %s \"\${$2}\""' _ "$OLD_CONTEXT" "$1"; }
OLD_MODEL_SHA=$(old_value V4_MODEL_SHA256)
OLD_CAL_SHA=$(old_value V4_CAL_SHA256)
OLD_STAGE2_SHA=$(old_value V4_STAGE2_SHA256)
[[ "$OLD_MODEL_SHA" == "$V4_MODEL_SHA256" ]] || { echo "ERROR: Stage1 model changed; full acceptance required." >&2; exit 4; }
[[ "$OLD_CAL_SHA" == "$V4_CAL_SHA256" ]] || { echo "ERROR: calibration changed; full acceptance required." >&2; exit 4; }
[[ "$OLD_STAGE2_SHA" == "$V4_STAGE2_SHA256" ]] || { echo "ERROR: Stage2 bundle changed; full acceptance required." >&2; exit 4; }

OLD_DEPLOY_SHORT=$(basename "$OLD_RUN_ROOT")
BACKUP=$(find "$V4_SESSION_ROOT_HOST/backups" -maxdepth 1 -type f -name "v4_code_${OLD_DEPLOY_SHORT}_*.tar.gz" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)
[[ -n "$BACKUP" && -s "$BACKUP" ]] || {
  echo "ERROR: could not find code backup for failed deployment $OLD_DEPLOY_SHORT" >&2
  exit 4
}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
tar -xzf "$BACKUP" -C "$TMP"

# These files determine Stage1/2/3 runtime behavior or the runtime/batch gate whose
# evidence is being reused. The hotfix is allowed to change only verifier/test tooling.
RUNTIME_FILES=(
  requirements.txt Dockerfile.v4_realtime
  precision_common.py voxel_common.py
  v4_realtime_core.py v4_realtime_pipeline.py v4_sparse_components.py
  v4_stage2_local.py v4_stage2_runtime.py v4_stage_contracts.py
  reconstruct_v4_stage3.py run_v4_realtime_session.py
  run_v4_stage1.py run_v4_stage2.py run_v4_stage3.py
  compare_v4_runtime_variants.py validate_v4_runtime_gate.py select_v4_runtime_mode.py
  benchmark_v4_stage1_batch_sizes.py select_v4_batch_size.py
)
for f in "${RUNTIME_FILES[@]}"; do
  [[ -f "$TMP/$f" && -f "$HERE/$f" ]] || { echo "ERROR: runtime identity file missing: $f" >&2; exit 4; }
  old=$(sha256sum "$TMP/$f" | awk '{print $1}')
  cur=$(sha256sum "$HERE/$f" | awk '{print $1}')
  [[ "$old" == "$cur" ]] || {
    echo "ERROR: runtime/gate source changed in $f; refusing to reuse old H100 gate." >&2
    echo "Run full acceptance instead: $HERE/run_v4_production_tests_on_nebius.sh" >&2
    exit 5
  }
done

echo "Runtime/model identity check: PASS"
echo "Prior failed test: $OLD_TEST_ROOT"
echo "Prior code backup: $BACKUP"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
REC_HOST="$V4_RUN_ROOT_HOST/tests/recovery_${STAMP}"
mkdir -p "$REC_HOST"
cp -f "$OLD_TEST_ROOT/runtime_variant_equivalence.json" "$REC_HOST/"
cp -f "$OLD_TEST_ROOT/v4_runtime_mode.env" "$REC_HOST/"
cp -f "$OLD_TEST_ROOT/v4_batch_size.env" "$REC_HOST/"
[[ -d "$OLD_TEST_ROOT/batch_sweep" ]] && cp -a "$OLD_TEST_ROOT/batch_sweep" "$REC_HOST/batch_sweep"
mkdir -p "$REC_HOST/prior_evidence"
cp -f "$OLD_TEST_ROOT/FAILED.txt" "$REC_HOST/prior_evidence/previous_FAILED.txt"
cp -f "$OLD_TEST_ROOT/replay_quick/COMPLETED.json" "$REC_HOST/prior_evidence/quick_COMPLETED.json"
for src in \
  "$OLD_TEST_ROOT/independent_stage1_relocated/STAGE1_COMPLETED.json" \
  "$OLD_TEST_ROOT/independent_stage2_relocated/STAGE2_COMPLETED.json" \
  "$OLD_TEST_ROOT/independent_stage3/STAGE3_COMPLETED.json" \
  "$OLD_TEST_ROOT/independent_stage3_repeat/STAGE3_COMPLETED.json"; do
  cp -f "$src" "$REC_HOST/prior_evidence/$(basename "$(dirname "$src")")_$(basename "$src")"
done
cat > "$REC_HOST/recovery_context.txt" <<EOF
recovery_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
current_deployment_fingerprint=$V4_DEPLOY_FINGERPRINT
old_test_root=$OLD_TEST_ROOT
old_run_root=$OLD_RUN_ROOT
old_code_backup=$BACKUP
model_sha256=$V4_MODEL_SHA256
calibration_sha256=$V4_CAL_SHA256
stage2_bundle_sha256=$V4_STAGE2_SHA256
runtime_source_identity=passed
reason=previous acceptance stopped only at stale replay metadata verifier key
EOF

REC=$(v4_to_container_output "$REC_HOST")
OLD_TEST=$(v4_to_container_output "$OLD_TEST_ROOT")
NAME=${NAME:-v4-recover-$(printf '%s' "$V4_SESSION_SAFE" | tr -cd 'A-Za-z0-9_.-' | cut -c1-36)-${V4_DEPLOY_SHORT:0:8}}
CMD="bash v4_recovery_after_verifier_hotfix_inside_docker.sh '$REC' '$OLD_TEST' '$V4_SESSION_FILTER' '$V4_MODEL' '$V4_CAL' '$V4_STAGE2' '$V4_DIAG_ROOT' '$V4_DEPLOY_FINGERPRINT'"
v4_detached_run "$NAME" "$REC/recovery.log" "$CMD"
printf '%s\n' "$REC_HOST" > "$V4_RUN_ROOT_HOST/LATEST_TEST_ROOT.txt"
printf '%s\n' "$REC_HOST" > "$V4_SESSION_ROOT_HOST/LATEST_TEST_ROOT.txt"
echo "Recovery test root: $REC_HOST"
echo "Monitor: $HERE/show_v4_production_state_on_nebius.sh"
