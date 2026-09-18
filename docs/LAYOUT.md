# Layout migration contract

Branch: `codex/v4-package-layout`. Baseline: `4e45f533021bdbe17a3c9227448e84e7236e15cf`.

`layout-map.json` records historical paths, new paths, and SHA-256 hashes of original files. `tests/test_layout.py` verifies every mapped computation module under `src/poleline` against those original hashes. No model operations, thresholds, geometry algorithms, or training calculations were changed by this reorganization. Performance experiments remain in the separate performance workspace.

## Compatibility changes

Historical Python paths are adapters that execute the original implementation while preserving its historical `__file__` for relative path discovery. A package import adapter supports canonical `poleline.*` names and historical names. Stage 1 model classes are checked for identity under both import orders, preserving historical checkpoint import names.

Shell entry points delegate to the moved scripts, which retain their historical location for path discovery. Docker launch scripts retain `/workspace/v4` and additionally mount the full checkout at `/workspace/poleline_repo` so adapters can locate `src/`. The code fingerprint and backup scripts include the moved source. Source-inspection validators resolve adapters to implementation files.

These adapters, deployment/path changes, package metadata, tests, and documentation are new or modified code. The entire repository is therefore not byte-identical; the migrated computation implementations are.

The stage1 preprocessing and stage2 geometry/line_graph/poles modules are thin API facades. Large original implementations remain intact; splitting their internals is a separate change.

## Supported execution

Use historical entry points or the editable package console commands. Moved benchmark/validation scripts that rely on historical relative imports should be invoked through their original `v4/` adapters. Keep the checkout intact, including `docs/layout-map.json`, `v4/`, and `ops/`. Standalone wheel installation is not supported by this transitional layout.

The generic V4 Stage 2 CLI and electrical V10 are distinct implementations. The migration does not change which implementation any historical command selects.

## Repeat validation

```sh
python tests/test_layout.py
python v4/validate_v4_production_source_contract.py
python tests/compare_layout_outputs.py --baseline /path/to/pristine/base \
  --data-root /path/to/stage2 --fixtures /path/to/mock/fixtures \
  --output artifacts/layout_validation/outputs
python benchmarks/mock_stage1.py --baseline-root /path/to/pristine/base
PYTHONPATH=v4 python ops/v4_stage1_inference_opt2/self_test_v4_stage1_opt2.py
```

The output comparison expects the provided 30-session Stage 2 dataset and explicitly marked synthetic refiner/calibration fixtures. The H100 comparison uses seeded random model weights and mock score calibration, BF16, batch 12, and 1/12/13/120 active cores. It checks exact score bytes and labels over repeated calls. Timings are recorded but are not output-equivalence fields.
