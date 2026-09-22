# V4 H100 optimization change summary

Branch: `harpreet/v4-h100-opt`  
Runtime commit: `76003562efa72d247667940e3c7e0168b5b1eedf`  
Upstream baseline: accepted E7, `fe11d435c2054650701bbe78c74675011278ccf3`

The changes reduce CPU preparation, tensor-copy and reconstruction overhead around V4 inference. They preserve the original repository layout. The model architecture, trained weights and decision thresholds are unchanged. Exact comparisons passed on the tested inputs; production acceptance remains outstanding because accepted production checkpoint, calibration and Stage 2 model assets were unavailable.

## Processing flow

```text
Input slice
  → read/decompress/parse and prepare sparse input
  → build core schedule and gather inputs
  → Stage 1 GPU inference
  → gather voxel predictions and write Stage 1 artifacts
  → Stage 2 connected components and geometric features
  → refinement and electrical-track reconstruction
  → write reconstruction outputs
```

Stage 1 can prepare future slices while the GPU processes the current slice, and write completed outputs on an ordered background writer. This is bounded lookahead within Stage 1. These changes do not implement overlapping Stage 2 reconstruction with the next Stage 1 inference.

## Modified existing code: seven files

| File | Change and purpose |
|---|---|
| [`ops/v4_stage1_inference_opt2/v4_realtime_core_opt2.py`](../ops/v4_stage1_inference_opt2/v4_realtime_core_opt2.py) | Integrates optional input-copy/layout optimizations and prepared core schedules into inference, reducing preparation on the critical path. |
| [`ops/v4_stage1_inference_opt2/run_v4_stage1_opt2.py`](../ops/v4_stage1_inference_opt2/run_v4_stage1_opt2.py) | Adds bounded input prefetch, scheduling ahead, and ordered asynchronous output writing. Exposes configuration and records preparation timing. |
| [`ops/v4_stage1_inference_opt2/profile_v4_stage1_opt2.py`](../ops/v4_stage1_inference_opt2/profile_v4_stage1_opt2.py) | Exposes inference optimization options to the existing profiling entry point. |
| [`ops/v4_stage1_inference_opt2/launch_v4_stage1_opt2_experiment.sh`](../ops/v4_stage1_inference_opt2/launch_v4_stage1_opt2_experiment.sh) | Passes optimization configuration through the experiment launcher. |
| [`ops/v4_stage1_inference_opt2/run_v4_stage1_opt2_experiment.sh`](../ops/v4_stage1_inference_opt2/run_v4_stage1_opt2_experiment.sh) | Wires optimization options into experiment execution. |
| [`ops/v4_stage2_stage1_electrical_v10_opt/v4_stage2_stage1_electrical_tracks_opt.py`](../ops/v4_stage2_stage1_electrical_v10_opt/v4_stage2_stage1_electrical_tracks_opt.py) | Accelerates line connectivity and voxel-support checks using batched operations. |
| [`v4/v4_sparse_components.py`](../v4/v4_sparse_components.py) | Groups component indices once instead of repeatedly scanning labels; reuses quantile work with precision-preserving dtype handling. Preserves component IDs and within-component source-row order. |

## New runtime helpers: seven files

| File | Purpose |
|---|---|
| [`v4/v4_groupnorm_layout.py`](../v4/v4_groupnorm_layout.py) | Combines GroupNorm input conversion and layout copying while retaining the normalization operation. |
| [`v4/v4_conv_layout.py`](../v4/v4_conv_layout.py) | Combines convolution input conversion and layout copying while retaining the convolution operation. |
| [`v4/v4_core_schedule.py`](../v4/v4_core_schedule.py) | Provides optimized CPU core-scheduling utilities. |
| [`v4/v4_input_prefetch.py`](../v4/v4_input_prefetch.py) | Prepares upcoming inputs on CPU workers with bounded lookahead. |
| [`v4/v4_output_writer.py`](../v4/v4_output_writer.py) | Writes completed outputs in order on a background worker and drains pending writes before completion. |
| [`v4/v4_input_pack.py`](../v4/v4_input_pack.py) | Checks tensor compatibility before dispatching the optional Kernel Factory input-copy kernel. |
| [`v4/v4_kernel_factory_input_copy.py`](../v4/v4_kernel_factory_input_copy.py) | Implements the retained Triton input-copy kernel. This is a copy/packing optimization, not a replacement model. |

These are runtime helpers. Locally generated mock fixtures, validation scripts, benchmarks, campaign inputs and profiling exports were excluded from the published optimization commit and its history. Existing upstream test/profiling files remain in the repository.

## Configuration and behavior

- New Stage 1 enable flags default to off. The latest experimental configuration enables input-copy fusion, prefetch with depth four and two workers, scheduling ahead, and asynchronous output writes.
- Scheduling ahead requires input prefetch. Lookahead limits queued slice count, not total memory in bytes.
- The optional Kernel Factory input-copy kernel is disabled in the latest measured configuration. No GroupNorm kernel from the later Kernel Factory campaign was integrated.
- Stage 2 changes require no new flags.
- Durable stage outputs and ordered completion remain part of the workflow.

See [H100_OPTIMIZATION_NOTES.md](H100_OPTIMIZATION_NOTES.md) for the full Stage 1 flag list and execution guidance.

## Measured results and validation scope

The latest component-extraction change produced the following incremental Stage 2 results. Both sides used the same mock Stage 2 refiners.

| Measurement | Before | After | Less wall time |
|---|---:|---:|---:|
| Full Stage 2 compute on 30 supplied prediction slices, excluding I/O | 165.62 ms/slice | 161.43 ms/slice | 2.5% |
| Full Stage 2 compute on eight fragmented mock prediction slices, excluding I/O | 2512.51 ms/slice | 2017.86 ms/slice | 19.7% |
| Cold Stage 2 CLI for eight mock slices, including startup and I/O | 23.761 s/batch | 19.757 s/batch | 16.8% |

Compute figures are means of per-slice medians from three timed repetitions after warmup. Cold CLI figures are medians of three fresh-process runs per variant. Fragmented mock outputs exaggerate the component-grouping benefit; the supplied-prediction gain is modest.

Exact checks covered 396 component-feature cases, 38 slice replays, six unprofiled and two profiled Stage 2 CLI executions, and structured contact/empty/singleton cases. Relocation, resume, and writer-failure checks also passed. These comparisons preserve numerical payloads; expected output-path and timing metadata differences are normalized.

An earlier Stage 1 lookahead comparison measured 891.13 → 867.13 ms per eight-slice mock sequence when increasing prefetch depth from two to four. That is an incremental throughput result, not single-slice latency or cumulative pipeline speedup.

These separate measurements must not be added into a full-pipeline latency claim. The complete production Stage 1 + Stage 2 + I/O target remains unverified. A historical separate-process output discrepancy in unchanged upstream E7 also remains unresolved; subsequent mock checks do not establish unconditional bitwise reproducibility.

## Documentation and local artifacts

The runtime publication also updated the Opt2 README and added `H100_OPTIMIZATION_NOTES.md`. This summary is documentation only; it introduces no further runtime changes.

Detailed local reports, CSVs, mock assets and Nsight captures remain under `Pole_Line_3D_Recon_h100_opt/perf/` on the workstation. They were not pushed. Local profiles 08 and 09 compare Stage 2 before/after the component changes; profile 07 remains the latest Stage 1 capture.
