# V4 vs V6.2 powerline recall analysis

## Why V4 can retain more legitimate line voxels

V4 achieved relatively high full-test line recall while accepting lower precision. V6.2 inherited V4 weights but added stronger geometry-aware training and Stage-2 component rejection. The result can be better precision yet visibly lower conductor recall.

The key failure modes identified in the V6.2 implementation were:

1. **V4 knowledge was initialization-only.** V6.2 copied V4 weights into the fine/coarse student, but no teacher/distillation loss prevented later fine-tuning from forgetting V4-supported line responses that disagree with imperfect GT.
2. **GT disagreement was only downweighted.** A V4-supported line voxel labelled background could receive less negative pressure, but it was not positively encouraged to remain a line.
3. **Line physics was partly axis-biased.** X/Y continuity terms represent axis-aligned wires more naturally than diagonal conductors.
4. **Stage-2 candidate extraction was too hard.** A single score threshold could break a sparse/occluded conductor into pieces or remove it entirely.
5. **Stage-2 hard geometry did too much rejection.** Short, slice-edge, sparse, or occluded pieces could be rejected before Stage 3 had access to session-level context.

## Implemented teacher-recall changes

### Positive-only V4 teacher

The frozen V4 network runs on the same local fine patch used by the V6.2 student. A GT-background voxel becomes eligible for teacher-positive line distillation only when:

- V4 line score is at least the configured teacher threshold;
- V4 line score exceeds V4 pole score by a margin; and
- local occupancy is not strongly vertical.

The teacher loss supervises the student line head, semantic line probability, and utility-objectness output. Teacher influence is confidence weighted. There is deliberately no V4 negative/background distillation term.

This makes the learning rule asymmetric: V4 can rescue plausible conductor signal, but V4 cannot force V6.2 to copy its background/negative errors.

### Rotation-friendlier local line support

The explicit horizontal support term now evaluates X, Y, +45-degree, and -45-degree XY directions with a small Z tolerance. Vertical support is evaluated separately. This is still local and uses no world/center coordinates.

### Class-specific calibration

Pole and line thresholds are selected against separate goals. The line score gives extra weight to recall, so calibration does not automatically erase the conductor sensitivity recovered from the teacher.

### Stage-2 hysteresis

Line candidates now use a strong seed threshold plus a lower connected-support threshold. Default values are:

- strong seed: 0.08
- weak connected support: 0.04
- line/pole competition ratio: 0.55
- minimum line component: 3 voxels

An isolated low-score voxel is still rejected. A weak voxel can survive only if it is connected to a component containing a strong line seed.

### Softer hard geometry

Stage-2 physical rejection is now reserved for clearly implausible geometry. Short or occluded fragments can survive to the learned refiner and then to Stage 3. Plausible GT disagreements remain ambiguous and are excluded from refiner-negative training.

## How to judge whether V6.2 recovered V4-like sensitivity

The package automatically compares strict-GT V4 and new V6.2 Stage-1 test metrics and writes:

- `training_diagnostics/v4_vs_v62_strict_gt_metrics.csv`
- `training_diagnostics/v4_vs_v62_line_metrics.png`
- `training_diagnostics/v4_vs_v62_line_metrics.json`

The most important first comparison is line recall, followed by visual inspection of recovered conductors that strict GT may call false positives. Precision should not be interpreted in isolation where GT omissions/misalignment are already known.

## Reconstruction retained in this package

The latest Stage-3 reconstruction also remains enabled:

- bounded pole heights;
- high-confidence completion of fragmented conductor paths between two poles;
- conservative hidden-pole inference from multiple independently pole-anchored, nonparallel approaching spans;
- rejection of hidden-pole creation from a lone partial conductor entering missing/uninteresting slices;
- no self-closing conductor loops or synthetic polygon/triangle completion;
- black poles and cyan conductors.
