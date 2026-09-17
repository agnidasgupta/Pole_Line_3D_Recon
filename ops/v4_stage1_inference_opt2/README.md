# V4 Stage 1 Opt2: production-equivalent inference plumbing

This directory tests computational-overhead reductions around the accepted V4
Stage 1 model. It does not change the inference procedure or redefine quality.

## Fixed production contract

Every run is hard-pinned to the original production settings:

- complete accepted `MultiHeadVoxelNet3D`, including every model head;
- accepted checkpoint and calibration;
- eager PyTorch execution (`torch.compile` is prohibited);
- `active_gpu`, BF16, batch size 12 and channels-last layout;
- fixed 64-cube input patches and 48-cube output cores;
- identical coordinate/distance inputs, score fusion, thresholds and serialization;
- no threshold tuning, relabeling, pruning, ONNX export or metric-driven changes.

The only experiment switch is `PINNED_D2H`. It changes how already-computed output
bytes are copied to host memory; it does not change model execution or scores.

`SCORE_ATOL=0` is enforced. The comparison is implementation equivalence—not a
quality metric. Incomplete dataset labels are not treated as ground truth: a V4
detection shown as a metric false positive may be a visually confirmed real pole
or line. Therefore metric artifacts are recorded only and never used to remove or
change production detections.

## Why the compile experiment was rejected

The compiler-enabled run completed all 162 slices and was faster, but changed a
production pole score by `0.0041994452476501465`. The result is rejected. The code
now prohibits compilation, batch changes, layout changes, head pruning and dummy
warm-up passes so that those variants cannot be launched accidentally.

## Install v3 on the existing experiment branch

On the Mac:

```bash
cd /Users/agni/Downloads

shasum -a 256 -c \
  V4_Stage1_Opt2_Production_Equivalent_v3.zip.sha256
```

Apply it to the existing branch and remove the obsolete compiler files:

```bash
REPO=/Users/agni/dev/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10
BRANCH=v4-stage1-inference-opt2
ZIP=/Users/agni/Downloads/V4_Stage1_Opt2_Production_Equivalent_v3.zip

git -C "$REPO" status --short
git -C "$REPO" switch "$BRANCH"

unzip -o "$ZIP" -d "$REPO"

git -C "$REPO" rm -f --ignore-unmatch \
  ops/v4_stage1_inference_opt2/Dockerfile.torch-compile \
  ops/v4_stage1_inference_opt2/build_v4_stage1_opt2_compile_image.sh

chmod +x "$REPO"/ops/v4_stage1_inference_opt2/*.sh

git -C "$REPO" diff --check
git -C "$REPO" status --short
```

Commit and push:

```bash
git -C "$REPO" add ops/v4_stage1_inference_opt2

git -C "$REPO" commit \
  -m "Restrict Stage1 Opt2 to production-equivalent inference"

git -C "$REPO" push origin "$BRANCH"
```

## Fetch v3 on Nebius

```bash
REPO=/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10
BRANCH=v4-stage1-inference-opt2

git -C "$REPO" status --short
git -C "$REPO" fetch github "$BRANCH"
git -C "$REPO" switch "$BRANCH"
git -C "$REPO" merge --ff-only "github/$BRANCH"

chmod +x "$REPO"/ops/v4_stage1_inference_opt2/*.sh
OPS="$REPO/ops/v4_stage1_inference_opt2"
```

The compiler derivative image is no longer required. Use the accepted image:

```text
va-v4-realtime:torch241-cu121
```

## E0: reproduce production without pinned D2H

Run one complete representative session:

```bash
export ONLY_GROUP_ID='VELASCO_CUT_CP/session1'
export EXPECTED_SESSIONS=1

VARIANT_NAME=e0_production_control \
IMAGE=va-v4-realtime:torch241-cu121 \
COMPILE_MODEL=0 \
COMPILE_MODE=default \
BATCH_SIZE=12 \
CHANNELS_LAST=1 \
PINNED_D2H=0 \
PRUNE_EMBEDDING_HEAD=0 \
WARMUP_ITERATIONS=0 \
SCORE_ATOL=0 \
bash "$OPS/launch_v4_stage1_opt2_experiment.sh"
```

Monitor:

```bash
bash "$OPS/monitor_v4_stage1_opt2_experiment.sh"
```

E0 must end with:

```text
PASS_PRODUCTION_OUTPUT_EQUIVALENCE
V4_STAGE1_OPT2_ALL_SESSIONS_PRODUCTION_EQUIVALENT_AND_OK
```

If E0 fails, stop. It means the existing outer Opt1 scheduling/workspace changes do
not reproduce production exactly and must be narrowed before testing anything else.

## E1: reuse a pinned final-output buffer

Only after E0 passes, change one transfer-plumbing variable:

```bash
VARIANT_NAME=e1_production_pinned_d2h \
IMAGE=va-v4-realtime:torch241-cu121 \
COMPILE_MODEL=0 \
COMPILE_MODE=default \
BATCH_SIZE=12 \
CHANNELS_LAST=1 \
PINNED_D2H=1 \
PRUNE_EMBEDDING_HEAD=0 \
WARMUP_ITERATIONS=0 \
SCORE_ATOL=0 \
bash "$OPS/launch_v4_stage1_opt2_experiment.sh"
```

Again require exact production equivalence. Compare E0 and E1 only after both pass:

```bash
python "$OPS/rank_v4_stage1_opt2_variants.py" \
  /path/to/e0_production_control_run \
  /path/to/e1_production_pinned_d2h_run \
  --output /home/agni/v4_stage1_opt2_production_equivalent_ranking.csv
```

Retain pinned D2H only if it is exactly equivalent and repeatedly faster in
`stage1_wall_ms`. Do not use metric precision/recall changes to judge it.

## Full 30-session confirmation

Only if E1 passes and is measurably faster:

```bash
unset ONLY_GROUP_ID
export EXPECTED_SESSIONS=30

VARIANT_NAME=e1_production_pinned_d2h_full30 \
IMAGE=va-v4-realtime:torch241-cu121 \
COMPILE_MODEL=0 COMPILE_MODE=default \
BATCH_SIZE=12 CHANNELS_LAST=1 PINNED_D2H=1 \
PRUNE_EMBEDDING_HEAD=0 WARMUP_ITERATIONS=0 SCORE_ATOL=0 \
bash "$OPS/launch_v4_stage1_opt2_experiment.sh"
```

Package only after the monitor prints `READY_TO_PACKAGE=YES`:

```bash
bash "$OPS/package_v4_stage1_opt2_results.sh"
```

The failed compiler runs may remain in their timestamped output directories; they
are isolated and are not eligible for packaging or promotion.
