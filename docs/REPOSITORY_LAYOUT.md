# V4 clean-layout maintenance contract

## Branch model

`v4-clean-v4-only-r2` starts from `v4` and remains mergeable with it. Until a
validated cutover, `v4` remains the accepted production branch. Merge future
accepted production changes from `v4` into this branch; do not maintain an
unrelated copy of production source.

The branch uses the directory pattern of a clean layout but contains no source
or documentation copied from `harpreet/v4-clean-root`.

## Directory contract

```text
v4/                         accepted production compatibility source
ops/v4*/                    accepted production compatibility operations
src/poleline/stage1/        new canonical Stage-1 modules
src/poleline/stage2/        new canonical Stage-2 modules
src/poleline/stage3/        new canonical Stage-3 modules
src/poleline/io/            new shared I/O modules
src/poleline/cli/           new command modules
ops/stage1/                 new Stage-1 operations
ops/stage2/                 new Stage-2 operations
ops/stage3/                 new Stage-3 operations
ops/pipeline/               cross-stage operations
ops/validation/             equivalence and regression gates
ops/experiments/            fetch/launch helpers for pinned experiments
experiments/<id>/           experiment record, source reference, results
```

Each new experiment must have a directory under `experiments/` with its
purpose, exact source commit, baseline, run configuration, acceptance rule,
timing summary, and small review diagnostics. Large output data, checkpoints,
NPZ files, bundles and profiler captures do not belong in Git.

Use `ops/experiments/fetch_experiment_worktree.sh` to materialize a complete
published experiment source branch into a separate Git worktree. Do not treat
an experiment record as a standalone runtime checkout.

V6.2 is not part of the maintained V4 workflow. It remains only where already
present in the `v4` parent history and must not be added to new V4 directories.

## Production contract

Preserve the accepted checkpoint, calibration, active-GPU BF16 batch-12 path,
full model heads, score fusion, thresholds, labels, Stage-2 V10 geometry and
electrical constraints, durable output structure, and Docker-only Nebius
execution. Unlabeled voxels are not evidence that correct pole/line predictions
are false.
