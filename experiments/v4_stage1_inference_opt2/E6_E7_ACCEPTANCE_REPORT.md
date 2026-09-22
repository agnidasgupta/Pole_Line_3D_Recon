# V4 Stage 1 Opt2 E6/E7 acceptance report

## Decision

E7 is accepted as the current Opt2 Stage 1 experimental candidate. It preserves
the accepted V4 inference procedure and passed the complete saved-production
equivalence gate:

- 30 sessions accepted;
- 3,738 slices completed;
- zero failed sessions;
- 30 production-equivalence reports passed with `SCORE_ATOL=0`;
- the same checkpoint, calibration, full five-head `MultiHeadVoxelNet3D`,
  active-GPU BF16 execution, batch size 12, channels-last layout, 64-cube
  patches, 48-cube cores, score fusion, thresholds, decisions and artifacts.

This is an implementation-equivalence result. It does not interpret incomplete
annotations as authoritative negatives; an unlabelled voxel may still be a real
pole or line. The protected `v4` production branch is unchanged.

## Full 30-session timing

| Mean per slice | Saved production baseline | E7 candidate | E7 change |
|---|---:|---:|---:|
| Baseline-comparable total | 379.022 ms | 203.350 ms | -175.671 ms, **46.35% faster / 1.864x** |
| Stage 1 prediction wall time | 282.852 ms | 109.138 ms | -173.714 ms, **61.42% faster / 2.592x** |

`baseline_comparable_total_ms` is CSV read plus sparse-item preparation plus
the Stage 1 prediction call. Artifact and manifest writes are recorded for
operations but are not included in that baseline comparison.

## E6 and E7 paired gates

E6 removed detailed CUDA-event timing only while retaining the already-exact
gather-buffer lifetime safeguard. It passed two exact paired gates. E7 added
only CUDA Graph capture/replay for the unchanged fixed-shape batch-12 model
forward and score-fusion work.

| Pair | Comparison | All slices | Capture-excluded steady state |
|---|---|---:|---:|
| 1 | E6 86.295 ms to E7 86.065 ms | E7 0.23 ms, **0.27% faster** | E6 82.497 ms to E7 81.765 ms; **0.89% faster** |
| 2 | E6 86.356 ms to E7 85.825 ms | E7 0.53 ms, **0.61% faster** | E6 82.227 ms to E7 81.596 ms; **0.77% faster** |

Both E7 runs had exact saved-production outputs with zero known-positive losses,
zero class flips and zero unverified additions. The confirmation run recorded
one CUDA Graph capture and 794 graph replays. The one-time capture is included
in the all-slice comparisons.

## Nsight Systems evidence

The accepted E7 capture profiled five measured passes after three warm-up
passes on slice 231 of
`NYSEGDistVegMgnt_AUBURN_-_SPRUCE_HAVEN_FARMS_TAP_520-3p_2029/session2`
(105,703 occupied rows). The measured mean wall time was 98.598 ms, with a
97.397 ms minimum and 100.410 ms maximum.

- Nsight recorded 30 `cudaGraphLaunch` calls, or six graph launches per
  measured pass, confirming replay of the captured fixed-shape batches.
- The five NVTX iteration ranges were 97.391–100.403 ms.
- `cudaMemcpyAsync` accounted for 423.210 ms across 125 API calls; its 9.083 us
  median and 84.750 ms maximum indicate host-side waiting/serialization in the
  observed API duration rather than a raw device-copy bandwidth limit.
- GPU memory operations totalled 1.284 ms device-to-device, 0.861 ms
  host-to-device and 0.156 ms device-to-host across the capture. Consequently,
  the large wall-clock `d2h_gather_ms` instrumentation value is not evidence of
  an isolated device-to-host bandwidth bottleneck.

The raw `.nsys-rep` and SQLite export remain outside Git. Reviewed text and JSON
summaries are sufficient for repository documentation.

## Next phase

E8 may now test asynchronous CSV read/sparse preparation, GPU inference and
durable write overlap. It must retain the accepted E7 procedure and write the
same `stage1`, `stage1_inference`, timing, status and manifest artifacts at the
same paths. E9 in-memory Stage 1 to Stage 2 handoff remains gated behind a
separate E8 exactness and full-session acceptance result.
