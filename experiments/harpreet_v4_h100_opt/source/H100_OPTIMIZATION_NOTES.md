# H100 Stage 1 and Stage 2 optimizations

Based on accepted upstream E7 (`fe11d435c2054650701bbe78c74675011278ccf3`). Original repository layout; package-layout branch unchanged. This branch contains runtime changes and documentation only. Locally generated mock fixtures, mock validation/benchmark scripts, Kernel Factory campaign inputs, datasets and binary profiles are excluded from this commit and its history.

## Runtime changes

- Stage 1: opt-in GroupNorm/Conv3d input-copy fusion, bounded CPU input preparation with lookahead, core scheduling ahead, and ordered asynchronous artifact writing.
- Optional guarded Kernel Factory input-copy kernel; disabled in the reported latest configuration.
- Stage 2: sparse line connectivity/support improvements, stable component index grouping and precision-preserving quantile reuse. Component IDs, source-row order, thresholds and geometry math are retained.
- Seven new runtime modules in `v4/`: `v4_conv_layout.py`, `v4_core_schedule.py`, `v4_groupnorm_layout.py`, `v4_input_pack.py`, `v4_input_prefetch.py`, `v4_kernel_factory_input_copy.py`, `v4_output_writer.py`. These are runtime helpers, not mock scripts.

## Running the experimental Stage 1 configuration

Use `ops/v4_stage1_inference_opt2/run_v4_stage1_opt2.py` with your normal input, output, model, calibration, session and timing arguments. Add:

```text
--cache_reference_coordinate_channels 1
--cache_coordinate_channels 0
--detailed_cuda_timing 0
--retain_gather_host_buffers 1
--use_cuda_graph 1
--channels_last_weights 1
--groupnorm_input_layout 1
--conv_input_layout 1
--prefetch_inputs 1
--prefetch_depth 4
--prefetch_workers 2
--async_output_writes 1
--prepare_core_schedule 1
--kernel_factory_input_pack 0
```

New Stage 1 enable flags remain off by default. Core scheduling ahead requires prefetch. Input lookahead bounds slice count, not bytes; size host memory accordingly. Async output writes are drained before completion. Runtime dependencies are the existing PyTorch/NumPy/SciPy/pandas environment; the optional CUDA input-copy helpers use Triton when enabled. Use the existing Stage 2 runner `ops/v4_stage2_stage1_electrical_v10_opt/run_v4_stage2_stage1_electrical_tracks_opt.py`; Stage 2 changes require no new flags.

## Validation and measurement limits

No accepted production checkpoint/calibration/Stage 2 bundle was available. Measurements use mock weights/refiners; supplied real predictions were also replayed through the same mock Stage 2 refiners on both sides. They do not establish production quality or the complete two-stage latency target.

Latest Stage 2 component change, three timed repetitions after warmup:

| Scope | Before | After | Less time |
|---|---:|---:|---:|
| 30 supplied prediction slices, full Stage 2 compute (no I/O) | 165.62 ms/slice | 161.43 ms/slice | 2.5% |
| 8 fragmented mock prediction slices, full Stage 2 compute (no I/O) | 2512.51 ms/slice | 2017.86 ms/slice | 19.7% |
| Cold Stage 2 CLI for 8 mock slices, startup and I/O included, median of 3 runs | 23.761 s | 19.757 s | 16.8% |

All tested component/full Stage 2 payloads matched exactly: 396 differential cases, 38 slice replays, six unprofiled and two profiled CLI executions, plus structured contact/empty/singleton, relocation, resume and writer-failure checks. Mock fragmentation exaggerates gains; the supplied-prediction gain is modest. Timings from different experiments must not be added into a pipeline latency claim.

Earlier Stage 1 mock sequence tests measured 891.13 to 867.13 ms for eight slices when increasing lookahead from two to four; this is throughput, not single-slice latency. Separate unchanged-upstream E7 processes once produced different scores/labels; later mock validation passed but the historical repeatability issue remains unresolved. Disabling cuDNN autotuning failed exact score equivalence and was rejected. Revalidate with accepted production assets before deployment.

Full local test reports, CSVs, mocks and Nsight captures remain on the workstation under `Pole_Line_3D_Recon_h100_opt/perf/`; they are intentionally not published.
