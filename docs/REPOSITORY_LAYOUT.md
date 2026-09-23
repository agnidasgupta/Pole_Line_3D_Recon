# V4-only clean repository layout

## Source provenance

Production content is copied from the owner-authored `v4` branch only. V4
experiment snapshots are copied only when their branch-only commits are
owner-authored; otherwise the experiment is catalogued without a source copy.
The
layout pattern is informed by `harpreet/v4-clean-root`, but no content is copied
from that branch. The sole external-source exception is the isolated
`harpreet/v4-h100-opt` experiment stored below `experiments/`.

## Layout

```text
src/poleline/
  stage1/       # future canonical Stage-1 modules
  stage2/       # future canonical Stage-2 modules
  stage3/       # future canonical Stage-3 modules
  io/           # future shared I/O modules
  cli/          # future canonical command modules
ops/
  stage1/       # future Stage-1 launch, monitor, profile, package scripts
  stage2/       # future Stage-2 launch, monitor, profile, package scripts
  stage3/       # future Stage-3 launch, monitor, profile, package scripts
  pipeline/     # future Stage-1 to Stage-3 orchestration
  validation/   # future equivalence and regression gates
experiments/
  <experiment-id>/
    README.md
    source/     # V4-only source snapshot or patch
    results/    # small, reviewable timings and diagnostics only
docs/
v4/             # accepted V4 compatibility source; unchanged until audited migration
ops/v4*/        # accepted V4 compatibility operations; unchanged until audited migration
```

## Rules

1. Put every new experiment in `experiments/<experiment-id>/`. Include its
   hypothesis, exact source revision, configuration, acceptance rule, timings,
   and small diagnostics. Do not put datasets, models, NPZ files or profiler
   captures in Git.
2. Put future operational scripts in the matching `ops/<stage>/` directory.
   Put cross-stage workflows in `ops/pipeline/` and gates in `ops/validation/`.
3. Put newly developed canonical code in the matching `src/poleline/<stage>/`
   directory. Do not move existing accepted V4 modules unless an import/CLI
   compatibility and production-equivalence gate accompanies the move.
4. Preserve the V4 production contract: checkpoint, calibration, BF16 batch 12,
   score fusion, thresholds, labels, geometry, electrical constraints, durable
   output structure, and Docker-only Nebius execution.
5. V6.2 is intentionally absent from this branch.
