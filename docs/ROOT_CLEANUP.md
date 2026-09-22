# Root cleanup — isolated branch

Branch: `harpreet/v4-clean-root`, based on optimized layout commit `2f4fd6a57bb34aff1bee6e8fcd4bbb6c964eec39`. This work does not merge into or update `main`, `harpreet/v4-package-layout`, `harpreet/v4-h100-opt`, or any other existing branch.

## What moved out of the root

- Removed 53 obsolete root-level Python/shell compatibility wrappers and historical-document symlinks. Their actual implementations/documents remain under `legacy/v62/` and `docs/archive/`.
- Moved the historical V6.2 Dockerfile and requirements into `legacy/v62/`. Root `requirements.txt` now selects `v4/requirements.txt`; package dependencies already used that V4 file.
- Adjusted legacy shell script path setup to run from `legacy/v62/`, where their sibling scripts, Python modules and Dockerfile now live.
- Retained `v4/` adapters and `ops/` aliases because current V4 imports and deployment launchers rely on them. Removing these is a separate migration.

[root-cleanup-map.json](root-cleanup-map.json) lists each old root path and its replacement. The layout map marks removed aliases explicitly rather than pretending the old paths still exist.

## Entry points

For optimized V4, install the editable checkout and use `poleline-stage1-opt2` and `poleline-stage2-electrical`. Alternatively:

```sh
PYTHONPATH=src python -m poleline.cli.stage1_opt2 --help
PYTHONPATH=src python -m poleline.cli.stage2_electrical --help
```

The existing `v4/` and `ops/` commands remain supported. Use the flags and validation limits in [OPTIMIZATION_INTEGRATION.md](OPTIMIZATION_INTEGRATION.md).

Historical V6.2 entry points are now explicit:

```sh
python legacy/v62/infer_v62_stage1_stage2.py --help
bash legacy/v62/run_v62_inference.sh
```

The inference command requires the historical model/calibration/data environment. Legacy deployment workflows retain their old infrastructure assumptions and are not production-validated by this cleanup. Root-level V6.2 commands intentionally no longer exist on this branch.

For V4 containers, follow [docker/README.md](../docker/README.md), using `v4/Dockerfile.v4_realtime`. For the archived V6.2 image, use `legacy/v62/` as the Docker build context. Bare `docker build .` no longer selects a historical V6.2 image.

## Checks

The cleanup leaves all V4 computation, optimized Python implementations, model weights and decision thresholds unchanged. Source-hash checks, import/CLI checks, shell syntax and stage-boundary validation check the retained paths. All seven layout tests passed, as did the V4 source-contract check and both optimized CLI help checks. The archived V6.2 inference CLI also loaded successfully. All 278 V4 source/support files under `src/`, `v4/`, `experiments/` and `scripts/` were verified unchanged from the optimized layout parent. Independent CPU processes matched all non-timing reconstruction outputs on 30 supplied slices and one full synthetic Stage 2 case with mock refiners. Generated payloads stay in ignored `artifacts/`.

No new GPU speedup or H100 execution claim is made. The prior H100 instance remains unavailable, and production assets remain outstanding.
