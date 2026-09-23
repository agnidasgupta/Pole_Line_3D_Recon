# Result: `harpreet_v4_h100_opt`

**Status: validated experiment; not a production merge.**

## Fixed contract

The accepted V4 checkpoint, calibration, active-GPU BF16 execution, batch size
12, channels-last layout, full model heads, score fusion, thresholds, label
decisions, V10 Stage-2 bundle/profile, geometry rules, and output hierarchy were
held fixed. Missing pole/line labels are not treated as proof that a prediction
is false.

## Stage 1 — validated H100 result

Representative steady-state runs, excluding the first slice, measured:

| Variant | Mean Stage-1 wall time |
|---|---:|
| Baseline r1/r2 | 81.34 ms/slice |
| H100 optimized r1/r2 | 72.12 ms/slice |

**Result: 9.22 ms/slice lower latency (11.3%).**

The gain came from bounded input prefetch, preparing active-core schedules ahead
of GPU execution, retaining coordinate/layout inputs, and ordered asynchronous
output writes. The checkpoint, architecture, BF16 batch-12 model execution,
score fusion, thresholds, and prediction labels were unchanged.

Equivalence: zero known-positive losses, zero known-positive class flips, zero
unverified additions, and byte-identical Stage-1 r2 NPZ handoff arrays.

## Stage 2 — validated deterministic result

Across a representative 157-slice session, baseline and optimized r1/r2 each
produced **1,256 byte-identical semantic reconstruction CSVs** after the loaded
pole and line `ExtraTreesClassifier` refiners were evaluated serially
(`n_jobs=1`). The trained trees, bundle, features, thresholds, geometry, and
output structure were unchanged.

Without pinning those refiners, parallel evaluation (`n_jobs=-1`) occasionally
changes the last printed decimal by one or two ULPs. That does not change any
threshold decision, component, label, geometry, attachment, or voxel support,
but fails a byte-exact serialization gate.

Published pre-H100 Stage-2 measurements reported 2.5% lower full compute on
30 supplied prediction slices and 19.7% lower compute on a deliberately
fragmented mock fixture. These are Stage-2-only figures and must not be added
to the Stage-1 H100 improvement.

## Acceptance

Accepted as a validated experiment with deterministic Stage-2 refiner
evaluation. Production promotion still requires a versioned deterministic
refiner launch option and a source-to-clean-layout mapping review.

