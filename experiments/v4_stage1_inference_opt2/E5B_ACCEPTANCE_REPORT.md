# V4 Stage 1 Opt2 E5b acceptance report

## Decision

E5b reference-coordinate caching passed the complete V4 production-preservation
gate and is accepted as the current Opt2 Stage 1 inference candidate.

- 30 sessions accepted;
- 3,738 slices completed;
- zero failed sessions;
- 30 production-equivalence reports passed with `SCORE_ATOL=0`;
- checkpoint, calibration, model heads, BF16 execution, batch size, input/core
  geometry, fusion, thresholds and serialization remained unchanged.

This is an implementation-equivalence result. It is not evaluation against the
incomplete annotation set, and it does not treat unlabelled voxels as negative.

## Matched timing gate

The decisive E0/E5b comparison used the same 157-slice session in consecutive
runs on the same H100 runtime:

| Mean per slice | E0 paired control | E5b repeat | E5b change |
|---|---:|---:|---:|
| Stage 1 prediction wall time | 88.496 ms | 87.328 ms | -1.168 ms, **1.32% faster** |
| GPU feature assembly | 2.344 ms | 1.047 ms | -1.297 ms, **55.3% faster** |
| GPU model and score fusion | 78.144 ms | 78.060 ms | -0.084 ms, effectively unchanged |

The end-of-slice `d2h_gather_ms` wall timer includes synchronization with queued
GPU work and must not be interpreted as an isolated device-to-host transfer.
Actual memory-copy timing is obtained from Nsight Systems.

## Method

E5b caches only immutable one-dimensional FP32 coordinate values. Cache misses
use the same batched arithmetic and operation order as production. Every batch
continues to use production `torch.cat`, production channels-last conversion and
a newly allocated final model input. E5b does not reuse E5's final input buffer.

The pre-run CUDA self-test checks complete E0/E5b input equality for all 405
possible active-core centers in the 400x400x200 grid, every partial fixed-batch
padding count and exact model outputs. Saved-production comparison remains the
authoritative gate.

## Nsight Systems capture

The accepted E5b path completed a CUDA-timed Nsight Systems capture using:

- a median discovered slice from the session that rejected E5;
- three warm-up iterations and five measured iterations;
- CUDA Profiler API capture boundaries around measured work only;
- CUDA, NVTX and OS-runtime tracing;
- the unchanged full five-head model;
- BF16, batch 12, active-GPU and channels-last execution.

The profiling output contains `PROFILE_SUMMARY.json`, `NSYS_STATS.txt`, model
inventory, profiler tool information, the capture log and the binary
`stage1_opt2_nsys.nsys-rep`. The raw binary report is retained outside Git and
downloaded separately for local Nsight Systems analysis. Small text/JSON
summaries may be committed when they contain no dataset, model or generated
per-voxel inference payloads.

## Artifact policy

The slim result package contains timing CSVs, metrics, manifests, completion
markers and production-equivalence reports. It excludes:

- Stage 1 NPZ artifacts;
- trained models and calibration bundles;
- raw datasets;
- per-voxel inference CSV.GZ files;
- raw Nsight `.nsys-rep` and SQLite files.

The raw Nsight archive remains a separate local diagnostic artifact and is not
part of the Git repository.
