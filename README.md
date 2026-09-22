# Pole_Line_3D_Recon

**Clean-root branch: `harpreet/v4-clean-root`.** The optimized Stage 1/Stage 2 code is retained. Old V6.2 entry points are under `legacy/v62/`; historical documents are under `docs/archive/`. Existing branches are unchanged. See [root cleanup and migration notes](docs/ROOT_CLEANUP.md).

V4 voxel inference and pole/power-line reconstruction. The original layout migration is based on `4e45f533021bdbe17a3c9227448e84e7236e15cf`; it now incorporates accepted E7 and the runtime optimizations published at `46ece117e7f9a2fc9f61bca47782e3166b44d991`.

## Layout

```text
src/poleline/
  stage1/       # Model, sparse inference, preprocessing facade
  stage2/       # Refiners, V10 reconstruction, geometry/graph facades
  stage3/       # Rolling multi-slice reconstruction
  training/     # Stage 1 and Stage 2 training, component mining
  io/           # Durable stage contracts
  cli/          # Original generic V4 command implementations
  pipeline.py
scripts/        # Deployment, full-dataset operations, validation
experiments/    # Opt1/Opt2, electrical V10, Unity integrations
benchmarks/     # Profiling, comparisons, mock H100 validation
configs/        # Configuration guidance; no invented production calibration
docker/        # Dockerfile alias and build instructions
tests/         # Existing smoke tests and migration equivalence checks
docs/          # Layout map, validation report, historical documentation
legacy/v62/    # Older root-level implementation
artifacts/     # Ignored generated results
v4/, ops/      # Compatibility entry points for existing commands
```

Unchanged moved computation files retain their original bytes. Optimized implementations match the published optimization source byte for byte, with original and current hashes recorded separately. Compatibility adapters preserve historical imports, CLI behavior, and location-based file discovery. See [layout details](docs/LAYOUT.md) and [validation results](docs/LAYOUT_VALIDATION.md).

## Use

Use an editable checkout in an environment with the appropriate PyTorch/CUDA build:

```sh
python -m pip install -e .
python v4/run_v4_stage1.py --help
python v4/run_v4_stage2.py --help
python v4/run_v4_stage3.py --help
```

`poleline-stage1`, `poleline-stage2`, `poleline-stage3`, and `poleline-session` expose those same generic V4 entry points. **The generic Stage 2 command does not select electrical V10.** Use `poleline-stage1-opt2` and `poleline-stage2-electrical` for the optimized entry points (or `PYTHONPATH=src python -m poleline.cli.stage1_opt2` / `poleline.cli.stage2_electrical`). See [integration details](docs/OPTIMIZATION_INTEGRATION.md) and the documented [Stage 1 Opt2](experiments/v4_stage1_inference_opt2/README.md) and electrical V10 Opt commands under `ops/v4_stage2_stage1_electrical_v10_opt/` for those experiments. Historical `ops/` paths remain aliases.

Use the original [V4 production guide](v4/README.md) for checkpoint, calibration, runtime, and deployment arguments. Existing defaults are retained. This package currently requires the source checkout; a standalone wheel is not supported.

## Data flow

1. Stage 1 reads voxelized slices, assembles occupancy/distance/coordinate patches, runs MultiHeadVoxelNet3D, and gathers per-voxel scores and labels into durable outputs.
2. Stage 2 builds components and features, applies the supplied refiners, and reconstructs poles and conductor geometry. The electrical V10 implementation uses Stage 1 electrical tracks.
3. Stage 3 joins already-acquired slices into a rolling world reconstruction, retaining the original 450 ft window contract.

The accepted checkpoint and calibration are external artifacts. Synthetic fixtures exercise the code but cannot establish production model quality or production output equivalence.

Historical root documentation is preserved in [the original README](docs/archive/README.original.md).
