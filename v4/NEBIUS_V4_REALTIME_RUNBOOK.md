# Nebius runbook — V4 realtime production candidate

This runbook is deliberately limited to the V4 realtime pipeline.

## 0. Production contract used throughout

Every arriving recording slice is processed in this order:

```text
new slice S
   |
   +--> Stage 1: slice S only
   |
   +--> Stage 2: slice S only
   |
   +--> commit Stage-2 objects + manifest row for S
   |
   +--> Stage 3: acquired rows only where S-9 <= slice_seq <= S
          - missing sequence numbers allowed
          - future slices forbidden
          - maximum span 450 ft
```

The production runner hard-fails if all-session reconstruction is requested, if Stage 3 is not run every slice, or if the rolling limits are changed from 9 increments / 450 ft.

---

## 1. Install directly on Nebius first — recommended

From the Mac:

```bash
cd ~/Downloads
scp Pole_Line_3D_Recon_V4_Realtime_Production_Candidate.zip nebius-va:/tmp/
ssh nebius-va
```

On Nebius:

```bash
cd /workspace/voxel_poleline

STAMP=$(date +%Y%m%d_%H%M%S)
DST=/workspace/voxel_poleline/Pole_Line_3D_Recon_V4_Realtime_Production_Candidate

if [ -d "$DST" ]; then
  mv "$DST" "${DST}_backup_$STAMP"
fi

unzip -o \
  /tmp/Pole_Line_3D_Recon_V4_Realtime_Production_Candidate.zip \
  -d /workspace/voxel_poleline

cd "$DST"
chmod +x *.sh *.py
```

Confirm:

```bash
pwd
ls -1
```

---

## 2. Build the V4-only H100 image

```bash
cd /workspace/voxel_poleline/Pole_Line_3D_Recon_V4_Realtime_Production_Candidate

./build_v4_realtime_image_on_nebius.sh
```

The default image is:

```text
va-v4-realtime:torch241-cu121
```

The image uses PyTorch 2.4.1 / CUDA 12.1 and provides the Python/SciPy/scikit-learn dependencies required by all three V4 stages.

Verify:

```bash
sudo docker images | grep 'va-v4-realtime'
```

---

## 3. Run the production preflight

```bash
./run_v4_realtime_preflight_on_nebius.sh
```

Required final line:

```text
V4_REALTIME_NEBIUS_PREFLIGHT_OK
```

The preflight checks:

- accepted V4 checkpoint and calibration;
- prepared train/validation manifests;
- Python compilation;
- Stage-1/Stage-2 slice-local source contract;
- 9-increment/450-ft Stage-3 rolling contract;
- GroupNorm/BatchNorm architecture guard;
- CPU/GPU coordinate-channel equivalence smoke math;
- sparse Stage-2 component logic;
- Stage-2 runtime schema;
- Stage-3 missing-sequence and future-slice exclusion behavior.

Do not continue if the preflight fails.

---

## 4. Run the four-way H100 Stage-1 equivalence/speed gate

Use enough real files to make the decision meaningful. Start with 32:

```bash
MAX_FILES=32 \
AMP=bf16 \
BATCH_SIZE=12 \
SCORE_TOL=1e-4 \
./run_v4_runtime_variant_gate_on_nebius.sh
```

Inspect:

```bash
RUN=/workspace/voxel_poleline/outputs/poleline_voxel_run_session_groups/v4_realtime

cat "$RUN/diagnostics/runtime_variant_equivalence.json"
cat "$RUN/diagnostics/v4_runtime_mode.env"
```

Load the selected mode into the current shell:

```bash
source "$RUN/diagnostics/v4_runtime_mode.env"

echo "V4_RUNTIME_MODE=$V4_RUNTIME_MODE"
echo "EVALUATE_ALL_CORES=$EVALUATE_ALL_CORES"
echo "GPU_COORD_CHANNELS=$GPU_COORD_CHANNELS"
```

Selection policy:

```text
active_cpu -> only if numerically equivalent to full_cpu within tolerance
              AND zero Stage-1 label mismatches

full_cpu   -> fallback/reference when the gate does not pass
```

`full_gpu` and `active_gpu` are diagnostic variants only; the automatic selector does not promote them.

**Do not train Stage 2 before this gate**. Stage-2 mining must use exactly the Stage-1 runtime selected here.

---

## 5. Benchmark Stage-1 batch sizes on the selected runtime

The old raw benchmark used batch 8. The production candidate tests larger H100 batches while requiring score/label equivalence.

```bash
RUNTIME_MODE="$V4_RUNTIME_MODE" \
BATCH_SIZES=8,12,16,20 \
MAX_FILES=8 \
AMP=bf16 \
SCORE_TOL=1e-4 \
./run_v4_stage1_batch_sweep_on_nebius.sh
```

Inspect:

```bash
BATCH_DIR="$RUN/diagnostics/batch_sweep_${V4_RUNTIME_MODE}"

column -s, -t "$BATCH_DIR/batch_size_summary.csv" | less -S
cat "$BATCH_DIR/batch_size_summary.json"
```

Choose the fastest batch size that has zero label mismatch against the reference and score differences within tolerance. Save it for the rest of the run, for example:

```bash
export V4_BATCH_SIZE=12
```

If 12 is the best safe value, leave it at 12. Do not increase batch size purely because memory is available if latency does not improve.

---

## 6. Train Stage 2 once using the exact selected Stage-1 runtime

Stage 1 is the accepted pretrained V4 checkpoint. This command performs Stage-2 mining on train/validation slices and trains the local ExtraTrees component refiners.

```bash
cd /workspace/voxel_poleline/Pole_Line_3D_Recon_V4_Realtime_Production_Candidate

BATCH_SIZE="${V4_BATCH_SIZE:-12}" \
AMP=bf16 \
RESUME=1 \
./run_v4_stage2_training_on_nebius.sh
```

The launcher automatically sources `v4_runtime_mode.env` unless you explicitly override the runtime flags.

Monitor:

```bash
sudo docker logs -f poleline-v4-stage2-training
```

Wait:

```bash
sudo docker wait poleline-v4-stage2-training
```

Expected exit code:

```text
0
```

Verify:

```bash
cat "$RUN/STAGE2_TRAINING_COMPLETED.json"
cat "$RUN/stage2_mining/COMPLETED.json"
cat "$RUN/stage2_refiner/local_refiner_metrics.json"
ls -lh "$RUN/stage2_refiner/local_refiner_bundle.joblib"
```

Important Stage-2 review files:

```text
stage2_mining/stage2_mining_runtime.csv
stage2_mining/target_counts.csv
stage2_refiner/local_refiner_metrics.json
stage2_refiner/pole_threshold_search.csv
stage2_refiner/line_threshold_search.csv
stage2_refiner/pole_feature_importance.csv
stage2_refiner/line_feature_importance.csv
```

---

## 7. List available recorded sessions

```bash
find /data/voxel_csv_combined \
  -mindepth 2 \
  -maxdepth 2 \
  -type d \
  -name 'session*_slice*' \
  -printf '%P\n' \
  | sed -E 's#/(session[0-9]+)_slice[0-9]+$#/\1#' \
  | sort -u \
  | less
```

Choose one session, for example:

```bash
export SESSION_FILTER="59768101-C4990BB-2026/session3"
```

---

## 8. Run a clean 5-slice realtime benchmark first

This exercises the actual persistent Stage-1 -> Stage-2 -> incremental Stage-3 path while keeping the test short.

```bash
SESSION_FILTER="$SESSION_FILTER" \
RUNTIME_MODE="$V4_RUNTIME_MODE" \
BATCH_SIZE="${V4_BATCH_SIZE:-12}" \
MAX_SLICES=5 \
AMP=bf16 \
./run_v4_realtime_benchmark_on_nebius.sh
```

The benchmark wrapper:

1. starts a clean replay (`RESUME=0`);
2. processes each slice sequentially;
3. runs Stage 3 after every slice in-process;
4. disables bulky row-level CSVs;
5. waits for the container;
6. verifies every rolling Stage-3 snapshot;
7. writes P50/P95 timing summaries.

The final output prints:

```text
V4_REALTIME_BENCHMARK_OK
Output: ...
Timing: .../realtime_timing_summary.txt
```

Inspect the printed output directory:

```bash
BENCH=/workspace/voxel_poleline/outputs/poleline_voxel_run_session_groups/v4_realtime/benchmarks/<printed-directory>

cat "$BENCH/REALTIME_REPLAY_VERIFICATION.json"
cat "$BENCH/realtime_timing_summary.txt"
column -s, -t "$BENCH/realtime_slice_timing.csv" | less -S
```

Do not proceed to a full session if verification does not say the replay passed.

---

## 9. Run a full recorded session as realtime

For a clean timing benchmark over the entire session:

```bash
SESSION_FILTER="$SESSION_FILTER" \
RUNTIME_MODE="$V4_RUNTIME_MODE" \
BATCH_SIZE="${V4_BATCH_SIZE:-12}" \
MAX_SLICES=0 \
AMP=bf16 \
./run_v4_realtime_benchmark_on_nebius.sh
```

This is the preferred performance measurement because it includes the changing Stage-3 history window as the session progresses.

For the persistent production-style replay location instead of a timestamped benchmark location:

```bash
SESSION_FILTER="$SESSION_FILTER" \
BATCH_SIZE="${V4_BATCH_SIZE:-12}" \
AMP=bf16 \
RESUME=1 \
./run_v4_realtime_session_on_nebius.sh
```

The gated Stage-1 runtime mode is automatically loaded if `v4_runtime_mode.env` exists.

Monitor:

```bash
sudo docker logs -f poleline-v4-realtime-session
```

Wait:

```bash
sudo docker wait poleline-v4-realtime-session
```

Expected exit code:

```text
0
```

Canonical production output directory:

```bash
SAFE=$(printf '%s' "$SESSION_FILTER" | tr '/ ' '__')
REPLAY="$RUN/replays/$SAFE"

echo "$REPLAY"
```

---

## 10. Verify the strict realtime/past-only contract

Use the container-side path form for the verifier:

```bash
REPLAY_CONTAINER="/outputs/poleline_voxel_run_session_groups/v4_realtime/replays/$SAFE"

REPLAY="$REPLAY_CONTAINER" \
./verify_v4_realtime_replay_on_nebius.sh
```

This checks, among other things:

- replay completion;
- unique/increasing slice sequences;
- expected Stage-2 object files;
- one Stage-3 snapshot per processed slice;
- Stage-3 completion/rule metadata;
- reconstructed conductor slice ranges do not precede `latest-9`;
- reconstructed objects do not contain a future slice;
- pole source-slice evidence remains inside the acquired past-only window.

Inspect:

```bash
cat "$REPLAY/REALTIME_REPLAY_VERIFICATION.json"
cat "$REPLAY/realtime_timing_summary.txt"
column -s, -t "$REPLAY/realtime_slice_timing.csv" | less -S
cat "$REPLAY/stage3_incremental/$SAFE/LATEST.json"
```

Every processed slice has its own Stage-3 snapshot under:

```text
stage3_incremental/<safe_session>/slice_<sequence>/
```

There is no production all-session finalize stage.

---

## 11. Timing interpretation

The main realtime metric is:

```text
end_to_end_update_ms
```

It is the time from reading an arriving CSV through completed rolling Stage-3 reconstruction for that update.

Also compare:

```text
Stage 1 wall and GPU model time
Stage 2 component time
Stage 2 refiner/parameterization time
Stage 3 incremental wall time
Stage 3 internal algorithm time
Stage 3 wrapper overhead
Stage 3 phase times
```

Use P50 and P95, not only the mean.

---

## 12. Package the compact review files for sharing

For the canonical replay:

```bash
SESSION_FILTER="$SESSION_FILTER" \
REPLAY="$REPLAY" \
./package_v4_production_review_on_nebius.sh
```

This creates:

```text
/tmp/v4_production_review_<safe-session>.tar.gz
```

The archive contains the runtime gate, Stage-2 metrics, replay verification, per-slice timing, timing summary, manifest, and latest Stage-3 reconstruction snapshot without the large model/cache files.

From the Mac, set the same session value and derive the archive-safe name again:

```bash
cd ~/Downloads

SESSION_FILTER="59768101-C4990BB-2026/session3"
SAFE=$(printf '%s' "$SESSION_FILTER" | tr '/ ' '__')

scp nebius-va:/tmp/v4_production_review_${SAFE}.tar.gz .
```

Upload this archive for review before promoting the candidate.

---

## 13. Download the larger set of important V4 results to the Mac

On the Mac, from the candidate source directory:

```bash
cd ~/dev/Pole_Line_3D_Recon_V4_Realtime_Production_Candidate

./download_v4_production_results_to_mac.sh
```

Default local destination:

```text
~/Documents/VEG_Data/POLE_Voxel/v4_realtime_production_review/
```

The download excludes model/checkpoint/joblib/NPZ binaries and bulky per-slice Stage-2 mining cache material, while retaining diagnostics, metrics, replay timing/manifests, and Stage-3 reconstruction CSV/PNG/audit outputs.

---

## 14. Files to share for the final production decision

The compact archive should contain the key items. Individually, the most useful are:

```text
diagnostics/runtime_variant_equivalence.json
diagnostics/v4_runtime_mode.env

diagnostics/batch_sweep_<mode>/batch_size_summary.csv
diagnostics/batch_sweep_<mode>/batch_size_summary.json

STAGE2_TRAINING_COMPLETED.json
stage2_mining/stage2_mining_runtime.csv
stage2_mining/target_counts.csv
stage2_refiner/local_refiner_metrics.json
stage2_refiner/pole_threshold_search.csv
stage2_refiner/line_threshold_search.csv
stage2_refiner/pole_feature_importance.csv
stage2_refiner/line_feature_importance.csv

replays/<session>/COMPLETED.json
replays/<session>/REALTIME_REPLAY_VERIFICATION.json
replays/<session>/realtime_slice_timing.csv
replays/<session>/realtime_timing_summary.txt
replays/<session>/realtime_timing_summary.json
replays/<session>/inference_manifest.csv
replays/<session>/stage3_incremental/<session>/LATEST.json
latest Stage-3 snapshot summary/CSVs/audits
```

Do not change Stage-1 weights, Stage-2 thresholds, or Stage-3 geometry solely to chase latency until these measurements are reviewed.
