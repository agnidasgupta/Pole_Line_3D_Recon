# Pole / Line 3D Reconstruction V4 — Production Accepted

V4 is the production implementation of the three-stage realtime pole and power-line reconstruction pipeline. Normal production operation performs inference and reconstruction only; Stage 1 is not retrained during runtime.

## Production status

Production acceptance completed on **2026-08-25** using Nebius H100 on session `59768101-C4990BB-2026/session3`.

Accepted deployment settings:

- Stage 1 runtime: `active_gpu`
- Stage 1 batch size: `12`
- AMP: `bf16`
- fixed final batch shape: enabled
- GPU coordinate channels: enabled
- active-core scheduling: enabled
- Stage 3 execution: in-process
- Stage 3 in-memory rolling cache: enabled
- maximum sequence gap: `9`
- slice spacing: `50 ft`
- maximum span represented by the rolling window: `450 ft`
- maximum observed slice centers in `[S-9,S]`: `10`

Deployment fingerprint accepted by the production gate:

`d9977c39c443f5fa14f802c21771a1f493a5e699e1b76362b42d325f54577b1c`

The full 113-slice production replay passed with **0 verifier errors and 0 warnings**.

## Three independently executable stages

| Stage | Durable input | Durable output | History allowed |
|---|---|---|---|
| Stage 1 | one raw slice CSV/CSV.GZ | `stage1_artifacts/**.npz`, metadata, `stage1_manifest.csv` | none |
| Stage 2 | saved Stage 1 artifacts | pole, line-segment and line-vertex CSVs plus manifests | none |
| Stage 3 | saved Stage 2 objects/manifests | rolling reconstruction snapshots | acquired rolling history only |

The realtime runner uses in-memory handoff where it reduces latency, but every stage retains a durable disk boundary. Stage 2 can therefore be rerun from saved Stage 1 results, and Stage 3 can be rerun from saved Stage 2 results without rerunning earlier stages.

## 450-ft Stage 3 contract

Each sequence increment represents 50 ft. For newest slice sequence `S`, Stage 3 may use only:

```text
S - 9 <= slice_seq <= S
```

Nine sequence increments correspond to 450 ft. If every sequence exists, the interval contains 10 observed slice centers. Missing sequence numbers are valid. Future slices and slices older than `S-9` are prohibited.

## Accepted Stage 1 runtime

`active_gpu` was compared with the original `full_cpu` reference on 32 files.

| Runtime | Mean speedup vs `full_cpu` | Stage 1 label mismatches | Stage 2 topology mismatches | Geometry max error | Refiner probability max error |
|---|---:|---:|---:|---:|---:|
| `active_cpu` | 4.96x | 0 | 0 | 0 | ~4.4e-16 |
| `full_gpu` | 2.86x | 0 | 0 | 0 | ~4.4e-16 |
| **`active_gpu`** | **12.35x** | **0** | **0** | **0** | **~4.4e-16** |

The optimized GPU path transfers sparse coordinates/distances once per slice, scatters them into a GPU-resident volume, extracts required patches on GPU, creates coordinate channels on GPU, evaluates only active output cores, keeps fused scores on-device through batching, and performs one final occupied-row result transfer back to CPU.

Batch size 12 is the accepted production batch. Batch sizes 16–32 produced small numerical score/refiner changes and therefore were not promoted despite slightly lower benchmark wall times.

## Full-session realtime timing

Measured across 113 slices:

| Phase | Mean | P50 | P95 |
|---|---:|---:|---:|
| CSV read | 45 ms | 42 ms | 80 ms |
| sparse input preparation | 15 ms | 12 ms | 31 ms |
| **Stage 1 total** | **218 ms** | **189 ms** | **389 ms** |
| **Stage 2 total** | **181 ms** | **185 ms** | **241 ms** |
| **Stage 3 total** | **1,198 ms** | **939 ms** | **2,599 ms** |
| **arrival-to-publish** | **1,672 ms** | **1,456 ms** | **3,002 ms** |

Stage 1 is no longer the principal latency bottleneck. It accounts for about 13% of mean arrival-to-publish latency, while Stage 3 accounts for about 72%.

### Stage 1 GPU timing

Mean GPU-path costs:

- host pin: ~0.61 ms
- sparse H2D: ~0.12 ms
- sparse scatter: ~0.40 ms
- GPU patch extraction: ~0.37 ms
- GPU feature assembly: ~2.67 ms
- GPU model: ~90.1 ms
- GPU gather: ~0.98 ms
- D2H result gather: ~12.9 ms

CPU patch construction and CPU host-batch packing are eliminated in the accepted runtime.

### Stage 3 timing

Mean Stage 3 algorithm costs:

- fragment joining: ~346 ms
- span completion (pre hidden-pole): ~326 ms
- hidden-pole inference: ~2 ms
- span completion after hidden-pole: 0 ms when no hidden pole is added
- chain build + pole attachment: ~439 ms
- output write: ~12 ms
- wrapper/cache overhead: ~46 ms

The strongest next performance opportunities are therefore Stage 3 fragment joining, span completion and chain construction/attachment.

## Stage 3 scaling with rolling-window size

Stage 3 mean latency by number of observed slice centers:

| Observed centers | Mean Stage 3 latency |
|---:|---:|
| 6 | 640 ms |
| 7 | 825 ms |
| 8 | 1,081 ms |
| 9 | 1,599 ms |
| 10 | 2,281 ms |

The increase at 8–10 centers shows that Stage 3 still performs work that scales with the size/complexity of the active rolling graph. This is the primary area for a V4.x optimization pass.

## Recommended next optimization experiments

These experiments should be performed **after tagging the accepted V4 baseline**, so any latency optimization can be regression-tested independently.

1. **Incremental Stage 3 candidate graph** — retain valid fragment-pair compatibility between updates and calculate only edges involving newly added or evicted slices.
2. **Incremental chain topology** — update only chains/components affected by added/removed fragments instead of rebuilding all chains in the rolling window.
3. **Cache span-completion search state** — invalidate only paths touching changed poles/tracks.
4. **Profile object-count scaling** — add per-update counts for fragments, candidate join edges, chains, poles and span candidates and correlate those counts with Stage 3 subphase latency.
5. **Optimize high-latency span completion** — the worst full-session updates spent about 2.0 s in span completion alone; instrument candidate/path counts before changing algorithms.
6. **Reduce D2H only if needed later** — Stage 1 D2H is ~13 ms and is no longer material compared with Stage 3, so this is lower priority.
7. **Do not promote larger Stage 1 batches without equivalence** — batches 16–32 were marginally faster but failed the strict score/refiner equivalence threshold.

Any Stage 3 optimization must retain all existing geometric/electrical restrictions and must pass the same full-session replay verifier before promotion.

## Nebius execution rule

Do not run project Python directly on the Nebius host. Python commands must run inside the V4 Docker image because packages such as NumPy, pandas, SciPy, scikit-learn and PyTorch are container dependencies.

The `*_on_nebius.sh` launchers derive code/data/session/output paths programmatically and persist resolved paths and deployment fingerprints. Long-running tests execute in detached Docker containers so SSH expiry does not destroy work.

## Production verification

The accepted full-session replay produced:

- `V4_PRODUCTION_ACCEPTANCE_OK`
- 113 Stage 3 snapshots for 113 timing/manifest rows
- 0 replay verification errors
- 0 replay verification warnings
- independent Stage 1, Stage 2 and Stage 3 replayability confirmed
- runtime/model identity guard passed
- 450-ft `[S-9,S]` contract passed

Use `show_v4_production_state_on_nebius.sh` to inspect deployment status and `package_v4_review_bundle_on_nebius.sh` to collect review diagnostics.

## Release workflow

The accepted production baseline should be committed to branch `v4` and tagged before additional Stage 3 latency experiments. A recommended tag is:

```text
v4.0.0-production
```

Future timing work should be developed as separate commits/branches, for example `v4-stage3-incremental`, and merged only after the same equivalence, independent-stage and full-session replay gates pass.
