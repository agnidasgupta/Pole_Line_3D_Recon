# Nebius H100 validated V4 optimization experiment

## Purpose and provenance

This experimental record is based on the clean repository layout branch
`v4-clean-root`, created from `harpreet/v4-clean-root`. It records the Nebius
H100 validation performed against the original-layout experiment
`harpreet/v4-h100-opt` at commit
`76003562efa72d247667940e3c7e0168b5b1eedf`.

The clean layout already contains the mapped optimized implementations and
compatibility entry points. This record does **not** claim that the clean-tree
implementation is byte-identical to that later original-layout commit without
an updated `docs/optimization-map.json` check. It records the tested behavior,
assets, acceptance gates, and deterministic Stage-2 runtime requirement.

## Fixed production contract

The validation did not change the V4 production model or reconstruction
contract:

- accepted checkpoint and calibration;
- active-GPU BF16 inference, batch size 12 and channels-last layout;
- full model heads, 64-cube patch and 48-cube output core;
- unchanged score fusion, thresholds, labels and durable output structure;
- accepted Stage-2 refiner bundle and electrical profile;
- unchanged electrical V10 geometry, support, and attachment rules.

The comparison contract is implementation equivalence, not evaluation against
incomplete labels. An unlabeled voxel is not evidence that a predicted pole or
line voxel is false.

## Assets used

| Asset | SHA-256 |
|---|---|
| Stage-1 checkpoint | `1b8b20c0bb2b52a1617555ed72c34311ba3839effd674bb2cac5273040d909ee` |
| Stage-1 calibration | `dea4829143f33d1f674176185ecd59df620c50a70488a83b0a2d6e17b81784e1` |
| Stage-2 local refiner bundle | `c451d501c3a3ccf7598ce8254d4807483afe3f41e4500be8a9abcf705843e72d` |
| Selected electrical profile | `de79c637e9d70ff0d39c2765b8b6514f08a8547374079cd1f9f803cb9879ca1d` |

## H100 results

### Stage 1

The baseline and optimized H100 repetitions passed production-equivalence
checks on the representative session. With the first slice excluded, 156
steady-state slices produced the following means:

| Variant | Stage-1 wall time | Result |
|---|---:|---|
| Baseline r1/r2 | 81.30 / 81.39 ms | PASS |
| Optimized r1/r2 | 72.11 / 72.13 ms | PASS |

This is approximately **11.3% lower Stage-1 wall time** for the tested
configuration. Known-positive losses, known-positive class flips and
unverified additions were all zero in the production-equivalence reports.

The Stage-1 gain comes from implementation work around the unchanged network:
bounded CPU input prefetch, preparation of the active-core schedule ahead of
GPU execution, retained coordinate/layout inputs, and ordered asynchronous
artifact writing. The model architecture, checkpoint, BF16 batch-12 forward
pass, score fusion, thresholds and predicted labels are not changed.

The H100 experiment reports **per-slice Stage-1 latency**, not a cumulative
end-to-end pipeline claim. The baseline r1/r2 mean was 81.34 ms and the
optimized r1/r2 mean was 72.12 ms after excluding the first slice. That is
9.22 ms saved per steady slice, or 11.3%.

### Stage 2

For the 157-slice representative session, baseline and optimized r1/r2 runs
were compared across 1,256 semantic reconstruction CSVs per repetition.

| Gate | Result |
|---|---|
| r1 semantic reconstruction CSVs | PASS: 1,256/1,256 byte-identical |
| r2 semantic reconstruction CSVs | PASS: 1,256/1,256 byte-identical |
| Baseline serial-refiner repeat | PASS |
| Optimized serial-refiner repeat | PASS |

## Deterministic refiner requirement

The pole and line refiners are `ExtraTreesClassifier` instances with 600 trees
and `n_jobs=-1`. With unconstrained parallel evaluation, unchanged baseline
runs occasionally produced one-ULP differences in serialized
`refiner_probability` values. The affected CSV rows, components, geometry and
acceptance decisions were otherwise identical.

For byte-identical Stage-2 production-equivalence validation and deployment,
deserialize the `pole_model` and `line_model` and set only their in-memory
`n_jobs` attributes to `1` before `predict_proba`. This changes evaluation
scheduling only; it does not modify the bundle, trees, features, thresholds,
geometry, Stage-1 prediction, or output structure. The serial runtime was
validated across full r1/r2 Stage-2 comparisons above.

Do not claim raw byte reproducibility for the bare unpinned refiner runtime.

## Timing interpretation

The optimized Stage-2 implementation reduces repeated component-label scans,
uses batched connectivity and support checks, and reuses component/quantile
work. The published pre-H100 experiment measured a 2.5% reduction for full
Stage-2 compute on 30 supplied prediction slices and a 19.7% reduction on
eight intentionally fragmented mock slices; those figures exclude I/O and are
not additive to the Stage-1 gain. The fragmented fixture exaggerates the
component-grouping benefit.

This H100 validation established exact output equivalence but did not record a
new aggregate Stage-2 speedup claim. The accepted report therefore makes no
combined Stage-1 + Stage-2 real-time percentage claim until a dedicated H100
Stage-2 timing aggregation is retained alongside the validation artifacts.

## Result-change significance

Under the accepted deterministic runtime, there were **no semantic result
changes**: Stage-1 input handoff arrays were byte-identical; Stage-2 r1/r2
semantic reconstruction CSVs were byte-identical; and all pole, line,
geometry, attachment and voxel-support outputs matched. The only observed
uncontrolled-runtime variation was a one- or two-ULP final-decimal change in a
small number of serialized `refiner_probability` values. It did not affect a
threshold decision, component, geometry or label, but it invalidated a strict
byte-comparison gate. Serial refiner evaluation removes that nondeterminism.

## Scope and next gate

This is an accepted experimental validation record, not a production-branch
merge. Before promoting the reorganized implementation, add a versioned
deterministic-refiner launch option or adapter, update the optimization mapping
to the exact source revision being promoted, and repeat the same production
asset gate. No generated inference data, model file or NPZ artifact belongs in
Git.
