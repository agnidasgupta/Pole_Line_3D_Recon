# H100 optimizations in the package layout

Target branch: `harpreet/v4-package-layout`. Source: published runtime-only optimization commit `46ece117e7f9a2fc9f61bca47782e3166b44d991`, based on accepted upstream E7. The source's generated mock scripts, fixtures, benchmark results and Nsight reports are not imported.

## Location of optimized code

```text
src/poleline/
  stage1/
    scheduling.py
    input_pack.py
    kernels/
      conv_layout.py
      groupnorm_layout.py
      kernel_factory_input_copy.py
  stage2/
    components.py              # Stable grouping and quantile reuse
    reconstruction.py          # Electrical connectivity/support improvements
  io/
    input_prefetch.py
    output_writer.py
  cli/
    stage1_opt2.py             # Optimized Stage 1 command adapter
    stage2_electrical.py       # Electrical Stage 2 command adapter
experiments/v4_stage1_inference_opt2/
  v4_realtime_core_opt2.py     # Accepted E7 + opt-in H100 execution
  run_v4_stage1_opt2.py
  ...                        # Updated upstream E7 launch/profiling support
v4/, ops/                    # Historical import/entry-point compatibility
```

The historical `v4/v4_*` helper names execute these canonical implementations. The Stage 2 historical module likewise resolves to `src/poleline/stage2/reconstruction.py`. Existing generic Stage 1/Stage 2 commands are not silently redirected to the experimental variants.

## Running

After `python -m pip install -e .`, use `poleline-stage1-opt2` and `poleline-stage2-electrical` with the same arguments as their historical runners. Without installation:

```sh
PYTHONPATH=src python -m poleline.cli.stage1_opt2 --help
PYTHONPATH=src python -m poleline.cli.stage2_electrical --help
```

Use the opt-in flag list in [H100_OPTIMIZATION_NOTES.md](H100_OPTIMIZATION_NOTES.md). New Stage 1 optimizations remain disabled by default. Stage 2 changes require no new flags. No new Stage 1/Stage 2 overlap or in-memory handoff is implemented here; this integrates the optimizations already measured on the original-layout branch.

The full checkout remains required. Docker launchers mount it at `/workspace/poleline_repo` in addition to historical `/workspace/v4` and experiment mounts, so compatibility adapters find their implementations.

## Provenance and validation

[optimization-map.json](optimization-map.json) records source paths, mapped paths, source commit and SHA-256 hashes. Imported Python computation and runner files are byte-identical to the published optimization source. Shell mount changes and documentation link adjustments are listed explicitly. `layout-map.json` retains original migration hashes and separately records current hashes for the two updated Stage 2 implementations.

Run `python tests/test_layout.py` to check original/current computation hashes, optimization provenance, compatibility paths, model class identity and shell syntax. `python v4/validate_v4_production_source_contract.py` checks the existing stage-boundary contract. Numerical comparison should use the optimized original-layout checkout as its reference, not the pre-E7 layout baseline.

See [OPTIMIZATION_CHANGE_SUMMARY.md](OPTIMIZATION_CHANGE_SUMMARY.md) for the source changes and measurements. This integration does not establish a new speedup or production acceptance. Production checkpoint, calibration and Stage 2 assets remain required for that validation; the historical Stage 1 repeatability caveat remains in force.

### Integration checks completed

- Local layout/provenance, model/helper import identity, shell syntax and production source-contract checks passed.
- Both new module command entry points expose the expected CLI arguments.
- Independent optimized-original/layout processes matched all non-timing electrical reconstruction outputs on 30 supplied slices and one complete synthetic Stage 2 case with mock refiners.
- Actual Stage 2 CLI written payloads matched on structured contact, empty and singleton cases: two poles, four tracks, two attachments, all 509 accepted line voxels preserved. Relocated artifacts, resume without CSV rewrites and writer failure without false completion passed.
- All imported Python files match the published optimization source exactly. The only imported shell differences add layout-required full-checkout Docker mounts.
- A fresh H100 Stage 1/Stage 2 comparison could not run: SSH to `dead-plum-squid` was reset and `brev ls` reported no active instance in the current organization. No new GPU performance or GPU execution-equivalence claim is made.

Generated comparison payloads remain local in `artifacts/optimization_integration/`; mock fixtures and temporary validation drivers are not added to the repository.
