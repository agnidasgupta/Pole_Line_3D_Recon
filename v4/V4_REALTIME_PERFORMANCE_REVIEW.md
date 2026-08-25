# V4 realtime performance review

This document is intentionally limited to the V4 model, V4 runtime packages, and the attached raw speed benchmark.

## Accepted Stage-1 model

The accepted V4 model remains `MultiHeadVoxelNet3D` with patch/core 64/48 and local coordinate + distance channels. The later optimized V4 retraining completed much faster, but its held-out metrics were slightly lower on most measures than the accepted V4 checkpoint. Therefore the production candidate keeps the accepted V4 weights and optimizes only runtime scheduling, data movement, Stage-2 local object filtering, and Stage-3 incremental reconstruction.


## Accepted V4 held-out baseline

The accepted V4 full-test calibration reported:

- pole precision 0.6649, recall 0.7098, IoU 0.5228;
- power-line precision 0.5975, recall 0.8133, IoU 0.5254.

These values are the accuracy reference for Stage-1 runtime-equivalence work. Runtime optimization should not change the checkpoint, calibration, or Stage-1 labels unless a separate model-quality experiment is explicitly approved.

## V4 raw speed benchmark

The attached 100-file raw benchmark measured the original V4 Stage-1 path at approximately:

- 1.890 s mean end-to-end per slice including the shared CSV read;
- 1.830 s mean excluding the shared CSV read;
- 0.355 s CPU patch construction;
- 0.703 s H2D;
- 0.604 s GPU model + score fusion/threshold;
- 0.029 s D2H;
- 0.108 s CPU dense scatter.

The benchmark therefore shows that the production problem is not just convolution math. Tiling, host-to-device transfer, and dense scatter are material contributors.

## Runtime changes being validated

1. **Active-core scheduling**: retain the exact 64^3 network patch and 48^3 output core, but skip output cores that contain no occupied source voxel.
2. **CPU coordinate channels by default**: Step 11 separates active-core scheduling from GPU coordinate generation. The production candidate defaults to the original CPU coordinate-channel math until the H100 four-way equivalence gate proves another path is numerically safe.
3. **Occupied-row D2H gather**: only score rows corresponding to occupied voxels are copied back from the GPU.
4. **Slice-local sparse Stage 2**: exact sparse 26-connectivity avoids a dense 400x400x200 component-label volume. The learned refiner uses only local component features.
5. **Incremental Stage 3**: only Stage 3 sees more than one slice. After slice S is committed, Stage 3 sees acquired rows with sequence in `[S-9, S]`; future rows and older rows are excluded. Missing sequence numbers are valid.
6. **Persistent process**: Stage 1 model, Stage-2 bundle, and the Stage-3 Python module remain loaded while a session is replayed, avoiding process/model startup on every slice.

## Promotion gates

- Stage-1 runtime variant must have zero label mismatches against `full_cpu` and score deltas within the configured tolerance on real H100 data.
- Stage 2 must be trained using the exact Stage-1 runtime variant selected for production.
- Stage 2 must remain strictly per-slice and must not use session/world/center/GT-overlap fields as model features.
- Stage 3 must pass the past-only rolling-window contract and must never use a future slice.
- Realtime promotion should use P50/P95 per-slice timings from a full recorded-session replay, not a one-patch microbenchmark.
