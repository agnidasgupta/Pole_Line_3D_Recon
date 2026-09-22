# V4 Stage 1 Opt2: production-preserving inference experiments

> Local H100 experiments: this checkout adds opt-in model input-copy fusion, bounded input prefetch, experimental core scheduling ahead, ordered asynchronous output writes, and an optional retained Kernel Factory packing kernel on top of accepted E7. These additions have mock-weight regression evidence, not production acceptance. Step 14 also found an unresolved separate-process output discrepancy in unchanged upstream E7; see the current report before interpreting exactness claims. Usage and validation scope: [H100_OPTIMIZATION_NOTES.md](../../docs/H100_OPTIMIZATION_NOTES.md).


This directory reduces execution overhead around the accepted V4 Stage 1 model.
It does not change the production model or redefine inference quality.

## Fixed production contract

- accepted checkpoint and calibration;
- complete `MultiHeadVoxelNet3D` and every trained head;
- active-GPU, BF16, batch 12 and channels-last;
- 64-cube input patches and 48-cube output cores;
- identical core order, model inputs, fusion, thresholds and serialization;
- no compilation, pruning, relabeling, ONNX export or warm-up;
- exact saved-production comparison with `SCORE_ATOL=0`.

Exact comparison is an implementation regression gate, not evaluation against
incomplete labels. An unlabelled voxel is not necessarily background. Visually
confirmed poles and lines that lack labels must not be removed by optimization.

## Accepted result and experiment history

E0 reproduced production exactly over 3,738 slices in 30 sessions:

```text
stage1_wall_ms:             282.852 -> 112.298 ms (60.30%, 2.519x)
baseline_comparable_total:  379.022 -> 209.723 ms (44.67%, 1.807x)
```

| Variant | Result | Decision |
|---|---|---|
| E0 optimized control | Exact on 3,738 slices / 30 sessions | Accepted control |
| E1 pinned D2H | Exact, slower overall | Rejected |
| E2 remove detailed CUDA events | Production score drift from unsafe asynchronous gather-index source lifetime | Rejected |
| E3 retain gather host buffers | Exact, no repeatable benefit | Rejected |
| E4 precomputed gather plans | Exact twice; mean 0.37% faster | Rejected as operationally insignificant |
| E5 coordinate/input cache | Representative exact; full gate drifted after 14 sessions | Rejected |
| E5b reference coordinate cache | Exact on 3,738 slices / 30 sessions; 1.32% faster in matched gate | Accepted Opt2 candidate |
| E6 E5b + timing off + safe gather lifetime | Exact twice | Required safety basis for E7 |
| E7 E6 + CUDA Graph replay | Exact on 3,738 slices / 30 sessions | Accepted Opt2 candidate |

E5 reduced the two-run representative mean from 121.648 to 119.954 ms, but its
30-session gate failed on session 15. One production pole score changed by
`0.00019707530736923218`. A fresh E0 run reproduced that complete 157-slice
session exactly, proving the failure belongs to E5 rather than the environment.
E5 must not be resumed or promoted.

E5b subsequently passed the session that rejected E5, a fresh matched timing
pair, CUDA-timed Nsight Systems profiling and the complete 30-session gate.

## Accepted E5b result

The decisive same-session consecutive comparison was:

| Mean per slice | E0 paired control | E5b repeat | E5b change |
|---|---:|---:|---:|
| Stage 1 wall time | 88.496 ms | 87.328 ms | -1.168 ms, **1.32% faster** |
| GPU feature assembly | 2.344 ms | 1.047 ms | -1.297 ms, **55.3% faster** |
| GPU model and score fusion | 78.144 ms | 78.060 ms | effectively unchanged |

The full acceptance gate completed all 3,738 slices in 30 sessions with zero
failed sessions and 30 exact saved-production equivalence reports. This is an
implementation result, not a quality judgment against incomplete labels.

The accepted E5b Nsight Systems run used three warm-up and five measured passes,
CUDA Profiler API capture boundaries, and CUDA/NVTX/OS-runtime tracing. The raw
`.nsys-rep` stays outside Git. See
[`E5B_ACCEPTANCE_REPORT.md`](E5B_ACCEPTANCE_REPORT.md) for the method, timing and
artifact policy.

## Why E5b

The accepted E0 profile measured approximately 5.175 ms of feature assembly per
profiled slice. Every batch recalculates deterministic x/y/z coordinate channels,
concatenates them with occupancy and distance, and then materializes a
channels-last tensor.

E5 changed coordinate generation and final input materialization together. E5b
keeps only the safe, narrower part of that idea. On a cache miss it creates FP32
coordinate lines with the exact production batch expression. It then retains
only those immutable one-dimensional values. Every batch still uses production's
`torch.cat` followed by production's channels-last conversion, and the final
model input is newly allocated rather than reused.

E5b preserves:

- every coordinate value and channel position;
- occupancy and normalized-distance values;
- production batch padding, `torch.cat`, channels-last conversion and active-core order;
- the complete model forward and all five heads;
- BF16 autocast, score fusion and output decisions.

The CUDA self-test checks exact E0/E5b input equality for all 405 possible
active-core centers in the 400x400x200 grid and every partial fixed-batch padding
count. It also checks exact model outputs. This self-test is necessary but does
not replace saved-production session gates.

The E7 regression check requires one CUDA Graph capture for the reusable
workspace and a replay on every inference call. A second call must report graph
reuse (`cuda_graph_captured=0`), not a second capture.

E0 remains the default. Accepted E5b is enabled only by
`CACHE_REFERENCE_COORDINATE_CHANNELS=1`; its E3, E4 and E5 experiment flags
remain disabled. E6 deliberately enables only E3's already-exact host-buffer
lifetime safeguard while removing detailed events.

## E6 and E7

E6 retries removal of detailed CUDA timing without repeating E2's asynchronous
buffer-lifetime error. It requires `RETAIN_GATHER_HOST_BUFFERS=1`, which keeps
the pinned gather-index sources alive until the existing end-of-slice CUDA
synchronization. It changes no inference values or operations.

E7 builds only on an exact E6. It captures the unchanged fixed-shape batch-12
model forward, all five heads and score-fusion operations in a CUDA Graph.
Variable-length occupied-row gathering and D2H transfer remain outside the
graph. E7 is enabled only by `USE_CUDA_GRAPH=1`; no eager fallback is allowed.

Run E6 and E7 strictly in the order documented in
[`E6_E7_RUNBOOK.md`](E6_E7_RUNBOOK.md). E7 completed that gate with 30 accepted
sessions, zero failures and 30 exact production-equivalence reports. See
[`E6_E7_ACCEPTANCE_REPORT.md`](E6_E7_ACCEPTANCE_REPORT.md) for paired timing,
full-gate and Nsight Systems evidence. E8 asynchronous I/O and E9 in-memory
Stage 1 to Stage 2 handoff remain separate experiments; they must preserve all
existing durable output paths and schemas.

See [`V4_STAGE1_INFERENCE_COMPLEXITY_AUDIT.md`](V4_STAGE1_INFERENCE_COMPLEXITY_AUDIT.md)
for model-level and future training-level findings.

## Install from Mac

```bash
cd /Users/agni/Downloads
shasum -a 256 -c V4_Stage1_Opt2_E6_E7_v10_1.zip.sha256

REPO=/Users/agni/dev/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10
BRANCH=v4-stage1-inference-opt2
ZIP=/Users/agni/Downloads/V4_Stage1_Opt2_E6_E7_v10_1.zip

git -C "$REPO" status --short
git -C "$REPO" fetch origin "$BRANCH"
git -C "$REPO" switch "$BRANCH"
git -C "$REPO" merge --ff-only "origin/$BRANCH"
unzip -o "$ZIP" -d "$REPO"
chmod +x "$REPO"/ops/v4_stage1_inference_opt2/*.sh

git -C "$REPO" diff --check
git -C "$REPO" diff --stat
git -C "$REPO" status --short
```

Review and push:

```bash
git -C "$REPO" add README.md ops/v4_stage1_inference_opt2
git -C "$REPO" diff --cached --check
git -C "$REPO" diff --cached --name-only
git -C "$REPO" commit -m "Accept production-equivalent Stage1 E5b optimization"
git -C "$REPO" push origin "$BRANCH"
```

## Fetch on Nebius

```bash
REPO=/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10
BRANCH=v4-stage1-inference-opt2

git -C "$REPO" status --short
git -C "$REPO" fetch github "$BRANCH"
git -C "$REPO" switch "$BRANCH"
git -C "$REPO" merge --ff-only "github/$BRANCH"

OPS="$REPO/ops/v4_stage1_inference_opt2"
chmod +x "$OPS"/*.sh
```

Stop if the worktree is dirty or the update is not a fast-forward.

## Gate 1: run E5b on the session that rejected E5

Python runs only inside Docker through this launcher:

```bash
ONLY_GROUP_ID='NYSEGDistVegMgnt_AUBURN_-_SPRUCE_HAVEN_FARMS_TAP_520-3p_2029/session2' \
EXPECTED_SESSIONS=1 \
VARIANT_NAME=e5b_reference_coordinate_cache_session2 \
IMAGE=va-v4-realtime:torch241-cu121 \
COMPILE_MODEL=0 COMPILE_MODE=default \
BATCH_SIZE=12 CHANNELS_LAST=1 PINNED_D2H=0 \
PRUNE_EMBEDDING_HEAD=0 WARMUP_ITERATIONS=0 \
DETAILED_CUDA_TIMING=1 RETAIN_GATHER_HOST_BUFFERS=0 \
PRECOMPUTE_BATCH_GATHER_PLANS=0 \
CACHE_COORDINATE_CHANNELS=0 \
CACHE_REFERENCE_COORDINATE_CHANNELS=1 \
SCORE_ATOL=0 RESUME=1 \
bash "$OPS/launch_v4_stage1_opt2_experiment.sh"
```

Monitor:

```bash
bash "$OPS/monitor_v4_stage1_opt2_experiment.sh"
```

For a one-session run, `READY_TO_PACKAGE=NO` is expected. Require:

```text
ACCEPTED=1 FAILED=0 PRODUCTION_EQUIVALENT=1
STATE=EXITED
V4_STAGE1_OPT2_ALL_SESSIONS_PRODUCTION_EQUIVALENT_AND_OK
```

Save and inspect the run:

```bash
E5B_ROOT=$(cat /home/agni/LATEST_V4_STAGE1_OPT2_RUN.txt)
printf '%s\n' "$E5B_ROOT" > /home/agni/V4_STAGE1_OPT2_E5B_SESSION2_ROOT.txt

grep -E \
  'variant_name=|detailed_cuda_timing=|retain_gather_host_buffers=|precompute_batch_gather_plans=|cache_coordinate_channels=|cache_reference_coordinate_channels=' \
  "$E5B_ROOT/RUN_INFO.txt"
```

Expected:

```text
detailed_cuda_timing=1
retain_gather_host_buffers=0
precompute_batch_gather_plans=0
cache_coordinate_channels=0
cache_reference_coordinate_channels=1
```

Any score or saved-output difference rejects E5b immediately. Do not change
`SCORE_ATOL` and do not resume the rejected E5 full-data root.

## Compare E5b with the exact E0 session control

```bash
HOST_OUTPUTS=/workspace/voxel_poleline/outputs
E0_ROOT=$(cat /home/agni/V4_STAGE1_OPT2_E0_SESSION2_CONTROL_ROOT.txt)
E5B_ROOT=$(cat /home/agni/V4_STAGE1_OPT2_E5B_SESSION2_ROOT.txt)
E0_C="/outputs${E0_ROOT#$HOST_OUTPUTS}"
E5B_C="/outputs${E5B_ROOT#$HOST_OUTPUTS}"

docker run --rm \
  --mount "type=bind,source=$OPS,target=/workspace/opt,readonly" \
  --mount "type=bind,source=$HOST_OUTPUTS,target=/outputs" \
  va-v4-realtime:torch241-cu121 \
  python /workspace/opt/rank_v4_stage1_opt2_variants.py \
    "$E0_C" "$E5B_C" \
    --group-id 'NYSEGDistVegMgnt_AUBURN_-_SPRUCE_HAVEN_FARMS_TAP_520-3p_2029/session2' \
    --output /outputs/v4_stage1_opt2_e0_e5b_session2_ranking.csv
```

Also compare the targeted timing:

```bash
echo '===== E0 ====='
grep -E '^(gpu_feature_assembly_ms|stage1_wall_ms|baseline_comparable_total_ms)' \
  "$E0_ROOT/STAGE1_TIMING_SESSION_AVERAGES.txt"

echo '===== E5B ====='
grep -E '^(gpu_feature_assembly_ms|stage1_wall_ms|baseline_comparable_total_ms)' \
  "$E5B_ROOT/STAGE1_TIMING_SESSION_AVERAGES.txt"
```

If E5b is exact but improves Stage 1 wall time by less than 1%, reject it as
operationally insignificant. If it saves at least 1%, repeat a fresh matched
E0/E5b pair before profiling or running all 30 sessions.

## Profile only a confirmed E5b candidate

Profile only after exact equivalence and a repeatable timing improvement:

```bash
E5B_ROOT=$(cat /home/agni/V4_STAGE1_OPT2_E5B_SESSION2_ROOT.txt)
PROFILE_IMAGE=va-v4-realtime:torch241-cu121-nsight
PROFILE_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
PROFILE_LOG="/home/agni/V4_STAGE1_OPT2_E5B_NSYS_${PROFILE_STAMP}.log"

nohup env \
  PROFILE_VARIANT=e5b \
  IMAGE="$PROFILE_IMAGE" \
  RUN_ROOT="$E5B_ROOT" \
  SESSION_FILTER='NYSEGDistVegMgnt_AUBURN_-_SPRUCE_HAVEN_FARMS_TAP_520-3p_2029/session2' \
  PROFILE_TIMEOUT_SECONDS=1800 \
  bash "$OPS/run_v4_stage1_opt2_nsight_profile.sh" \
  > "$PROFILE_LOG" 2>&1 < /dev/null &

PROFILE_PID=$!
printf '%s\n' "$PROFILE_PID" > /home/agni/LATEST_V4_STAGE1_OPT2_E5B_PROFILE_PID.txt
printf '%s\n' "$PROFILE_LOG" > /home/agni/LATEST_V4_STAGE1_OPT2_E5B_PROFILE_LOG.txt
```

Success requires `NSIGHT_PROFILE_OK` and a nonempty pointer at:

```text
/home/agni/LATEST_V4_STAGE1_OPT2_E5B_PROFILE_ARCHIVE.txt
```

## Full 30-session gate

Run only after exact failing-session reproduction, at least 1% repeatable timing
benefit, and a successful E5b profile:

```bash
unset ONLY_GROUP_ID

EXPECTED_SESSIONS=30 \
VARIANT_NAME=e5b_reference_coordinate_cache_full30 \
IMAGE=va-v4-realtime:torch241-cu121 \
COMPILE_MODEL=0 COMPILE_MODE=default \
BATCH_SIZE=12 CHANNELS_LAST=1 PINNED_D2H=0 \
PRUNE_EMBEDDING_HEAD=0 WARMUP_ITERATIONS=0 \
DETAILED_CUDA_TIMING=1 RETAIN_GATHER_HOST_BUFFERS=0 \
PRECOMPUTE_BATCH_GATHER_PLANS=0 \
CACHE_COORDINATE_CHANNELS=0 \
CACHE_REFERENCE_COORDINATE_CHANNELS=1 \
SCORE_ATOL=0 RESUME=1 \
bash "$OPS/launch_v4_stage1_opt2_experiment.sh"
```

Promotion requires exact saved-production equivalence for all 30 sessions and
all 3,738 slices. Any mismatch rejects E5b.

## Package compact E5b results

The packager requires the successful full-run and profile pointers. It creates a
sub-100 MiB archive containing only timing CSVs, metrics, manifests, completion
markers, equivalence reports and text/JSON Nsight summaries:

```bash
RUN_ROOT=$(cat /home/agni/V4_STAGE1_OPT2_E5B_FULL30_ROOT.txt)
PROFILE_ROOT=$(cat /home/agni/LATEST_V4_STAGE1_OPT2_PROFILE.txt)

RUN_ROOT="$RUN_ROOT" \
PROFILE_ROOT="$PROFILE_ROOT" \
bash "$OPS/package_v4_stage1_opt2_results.sh"
```

Success requires:

```text
PACKAGE_OK
ACCEPTED=30 FAILED=0 PRODUCTION_EQUIVALENT=30 SLICES=3738
```

The archive deliberately excludes NPZ, model/checkpoint, raw dataset,
per-voxel inference CSV.GZ, `.nsys-rep` and SQLite files. The raw Nsight archive
is downloaded separately and should not be committed to Git.

## GitHub profiling artifacts

Commit this README and `E5B_ACCEPTANCE_REPORT.md`. Small text/JSON reports such
as `NSYS_STATS.txt` and `PROFILE_SUMMARY.json` may also be committed after
reviewing them for environment-specific paths. Do not commit:

- `stage1_opt2_nsys.nsys-rep` or Nsight SQLite exports;
- Stage 1 NPZs or per-voxel inference CSV.GZ files;
- model/checkpoint/calibration files;
- complete generated run directories or result archives.
