# Clean-root branch validation on L40S

Tested clean source: `fae3c0d7fe6efcada6d3a8e915deb60a01a5eff5`. Reference: original-layout optimized publication `46ece117e7f9a2fc9f61bca47782e3166b44d991`. This comparison checks layout/cleanup equivalence, not speedup over Agni's baseline.

Host: Brev `frightened-harlequin-crane`, NVIDIA L40S (46068 MiB), driver 565.57.01, eight host CPUs. Container: PyTorch 2.4.1+cu121, NumPy 2.1.1, pandas 3.0.6, SciPy 1.17.1, scikit-learn 1.9.1. Nsight Systems 2024.5.1. No H100 performance comparison is inferred.

## Result

**All tested outputs matched exactly.** Three fresh-process repetitions per implementation ran the actual Stage 1 and Stage 2 CLIs on the same eight supplied input slices. All 12 stage executions passed: 73 Stage 1 payload signatures and 92 Stage 2 signatures per execution. The additional two profiled clean-branch executions also matched the unprofiled clean run. Expected root paths, timing/progress metadata are normalized; no numerical tolerance is used.

All seven layout tests and the production source-contract check passed in the L40S container. The published computation files are unchanged by this test.

Test assets were explicitly synthetic: seed-42 model initialization with the real architecture, mock calibration and mock 600-tree Stage 2 refiners trained using repository training code. Missing original distance features retain the existing zero default. This is not production-quality validation. The historical upstream Stage 1 repeatability finding remains unresolved despite these successful runs.

Stage 1 used reference coordinate caching, CUDA Graphs, retained gather buffers, input-copy layout options, prefetch depth four with two workers, scheduling ahead and ordered asynchronous output writes. Detailed CUDA event timing and optional Kernel Factory input packing were disabled, matching the latest measured configuration.

## Timing observations

Median harness wall times for an eight-slice stage batch:

| Implementation | Stage 1 | Stage 2 |
|---|---:|---:|
| Original-layout optimized reference | 4.015 s | 14.966 s |
| Clean package layout | 4.066 s | 15.160 s |

These include fresh process startup, model/bundle loading, I/O, teardown and output-signature hashing by the validation harness. They are validation-run observations, not isolated inference timings or a controlled speedup claim. Reference runs preceded clean runs in each repetition. Small differences here do not establish a layout performance regression.

The clean-run timing CSVs report mean `stage1_wall_ms` of 268.83 ms over 24 slice executions, including first-slice graph capture; mean graph-capture contribution over all slices is 82.53 ms. Stage 2 reports mean `slice_wall_ms` of 1623.74 ms, including 1144.19 ms component extraction and 305.46 ms electrical-track work. Fragmented random predictions produce an unrepresentative reconstruction workload. Async Stage 1 per-slice fields must not simply be summed to infer batch wall time.

Execution is Stage 1 for the eight-slice batch, then Stage 2 for that batch. It is not an interleaved per-slice production pipeline. No 150/250 ms production target is demonstrated.

## Combined native profile and CSVs

Local artifacts are under `Pole_Line_3D_Recon_clean/artifacts/l40s_validation/`:

- `10_l40s_clean_stage1_stage2_mock.nsys-rep`: combined capture of the clean branch's latest measured Stage 1 and Stage 2 configuration, including durable I/O. CUDA graph-node tracing is enabled.
- Adjacent `.sqlite`: exported trace for analysis.
- `visual/stage12_timeline.png` and `.svg`: stage ranges, component/refiner/reconstruction lanes and recorded GPU work.
- `visual/stage12_profile_ranges.csv`: profile range durations.
- `execution_comparison.csv`: all unprofiled validation execution records.
- `l40s_clean_validation/*/stage*/timing.csv`: per-slice timing data.
- `l40s_clean_profile/stats_*.csv`: NVTX, CUDA API, kernel and memory summaries.

The trace covers 18.674 s from the first Stage 1 NVTX range start to the final Stage 2 range end. Stage 1's outer range is 2.999 s; Stage 2's is 13.932 s. The intervening time includes worker startup/imports and orchestration. Recorded GPU activity union is 1.651 s; it is not SM occupancy or whole-device utilization. GPU inactivity during the CPU Stage 2 batch is expected. Captured durations include profiling overhead and do not replace unprofiled measurements.

Native report SHA-256: `3c39abdbd9b97b700585075c9c416ca7f20549caa822a13e19b06388d640b1dd` (verified after download).

Only this Markdown report is published. Mock assets, temporary drivers, raw outputs and binary profiling artifacts remain outside Git. Existing branches remain unchanged; documentation is added only to `harpreet/v4-clean-root`.
