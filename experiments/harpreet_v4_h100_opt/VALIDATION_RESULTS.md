# H100 V4 optimization validation result

## Scope

This record covers the V4-only source snapshot from
`harpreet/v4-h100-opt` and the production-asset validation performed on a
Nebius H100. It is an experiment, not a production merge.

## Fixed contract

The accepted V4 checkpoint, calibration, active-GPU BF16 execution, batch size
12, channels-last layout, full model heads, score fusion, thresholds, label
decisions, V10 Stage-2 bundle/profile, geometry rules, and durable output
structure were held fixed. Known labels are incomplete: a voxel without a pole
or line label is not evidence that a correct pole/line prediction is false.

## Stage 1 measured H100 result

With the first slice excluded, the representative 156-slice steady-state runs
measured:

| Variant | Mean Stage-1 wall time |
|---|---:|
| Baseline r1/r2 | 81.34 ms |
| H100 optimized r1/r2 | 72.12 ms |

That is **9.22 ms, or 11.3%, lower Stage-1 latency**. The gain was obtained by
bounded input prefetch, preparation of active-core schedules ahead of GPU
execution, retained coordinate/layout inputs, and ordered asynchronous output
writing. It did not change model architecture, checkpoint, BF16 batch-12 model
execution, score fusion, thresholds or prediction labels.

Stage-1 production-equivalence results: zero known-positive losses, zero
known-positive class flips, zero unverified additions, and byte-identical
Stage-1 r2 NPZ handoff arrays.

## Stage 2 validation result

The optimized Stage-2 source reduces repeated component-label scans and uses
batched connectivity/support checks. Published pre-H100 measurements reported
2.5% lower full Stage-2 compute over 30 supplied prediction slices and 19.7%
on a deliberately fragmented mock fixture; these are not combined with the
Stage-1 result and are not a new aggregate H100 claim.

Across a 157-slice representative session, baseline and optimized Stage-2 r1
and r2 each produced **1,256 byte-identical semantic reconstruction CSVs**
after deterministic refiner scheduling was applied.

The existing pole and line refiners are 600-tree `ExtraTreesClassifier` models
with `n_jobs=-1`. Unpinned evaluation can produce one/two-ULP differences in
the final printed `refiner_probability` decimal. Those differences did not
change any threshold decision, component, label, geometry, attachment or voxel
support result, but do not meet a byte-exact gate. Setting the in-memory
`pole_model.n_jobs` and `line_model.n_jobs` to `1` after bundle load removes
that scheduling nondeterminism without changing the trained trees, bundle,
features, thresholds or output structure.

## Acceptance

The experiment is accepted as a validated experiment with deterministic Stage-2
refiner evaluation. Production promotion requires a versioned deterministic
refiner launch option and an exact source-to-clean-layout mapping check.
