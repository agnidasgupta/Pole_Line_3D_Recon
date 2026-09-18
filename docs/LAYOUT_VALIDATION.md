# Layout output-equivalence validation

Validated 2026-09-18 against pristine upstream commit `4e45f533021bdbe17a3c9227448e84e7236e15cf`.

| Check | Result |
| --- | --- |
| SHA-256 of 17 migrated computation modules under `src/poleline` | All identical to original files |
| Migration tests: hashes, aliases, model import identity in both orders, shell syntax | 4/4 passed |
| Existing V4 smoke tests | 8/8 passed |
| Production source contracts | Passed |
| V10 reconstruction self-test | Passed |
| H100 Opt2 CUDA self-test | Passed |
| Historical `/workspace/v4` Docker mount with full checkout mount | Passed |
| Real line reconstruction: one slice from each of 30 supplied sessions | All non-timing outputs exactly identical |
| Full Stage 2 synthetic fixture (pole/refiner/line path) | All non-timing outputs exactly identical |
| Stage 1 H100: 1, 12, 13, 120 active cores, 5 repetitions each | Scores byte-identical; labels identical; repeated baseline stable |

## What was compared

Stage 2 runs execute the pristine and reorganized sources in separate Python processes. Comparisons cover all returned non-timing fields: NumPy dtype, shape and bytes; DataFrame CSV serialization; exact floating-point values; geometry; identifiers and ordering. The real-slice cases exercise line reconstruction without accepted poles. The synthetic fixture exercises full Stage 2, including refiners and poles, using explicitly marked mock artifacts.

Stage 1 runs on an NVIDIA H100 PCIe, PyTorch 2.4.1+cu121, CUDA 12.1, cuDNN 90100. Both implementations use the same seeded MultiHeadVoxelNet3D (341,518 parameters), synthetic occupied voxels and mock calibration, BF16, batch 12. Pole, line, semantic and objectness arrays are compared by dtype, shape and bytes, and thresholded labels are compared exactly. The original and moved Opt2 implementations have separate workspaces. The shared model implementation is covered by the original-source hash check.

## Limits

This establishes exact agreement for the exercised cases. It does not establish end-to-end equivalence with the accepted production checkpoint/calibration, which were unavailable. It does not certify all deployment environments or serialized production checkpoints. Stage 3 passed its existing incremental/contract smoke tests but was not included in a separate before/after real-data comparison. Timing and environment metadata are intentionally excluded from output equality.

The source computation is byte-identical; import/path adapters, launch scripts, packaging, documentation and validation tooling changed. No performance optimization is included in this layout branch.

## Evidence and reproduction

Local generated evidence is under `artifacts/layout_validation/` (ignored by Git): `stage1_h100.json`, `stage1_h100.csv`, `outputs/before.json`, `outputs/after.json`, `outputs/summary.json`, `smoke_results.json`, `opt2_self_test.log`, `docker_legacy_mount.log`, and `cli_results.json`.

The initial local Opt2 self-test entry in `smoke_results.json` records a missing import path. Running with `PYTHONPATH=v4` then required CUDA; the H100 rerun completed successfully, as recorded in `opt2_self_test.log`.

See [reproduction commands](LAYOUT.md#repeat-validation) and [the original file/hash mapping](layout-map.json).
