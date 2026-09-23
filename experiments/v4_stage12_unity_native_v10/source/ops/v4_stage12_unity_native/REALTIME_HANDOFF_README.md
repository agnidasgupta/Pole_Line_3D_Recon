# V10 Unity Sentis Stage 1 + Native Stage 2 Runtime Handoff

## Scope

This archive is a self-contained Linux x86_64 runtime handoff for real-time
testing on an NVIDIA T4/T6-class Unity server:

- Stage 1 runs the accepted FP32 ONNX model using Unity 6.3 LTS and Sentis
  2.6.1 `GPUCompute`.
- Stage 2 runs in native C# in the same Unity process and consumes the Stage 1
  result directly in memory.
- Stage 2 uses the accepted ExtraTrees asset and the quality-equivalent
  voxel-supported reconstruction rules.
- Runtime does not require Python, NumPy, pandas, sklearn, joblib, Docker,
  Nebius, or network access.
- FP16 is included only for traceability. Production execution remains FP32
  until real-slice FP16 parity passes.

## Archive layout

```text
runtime/Linux/              Complete Unity Linux player and required libraries
model_assets/               ONNX, sidecar, and Stage 2 tree assets for audit/rebuild
scripts/                    Portable launch and monitor helpers
source/ops/...              Exact Unity-native source and build tooling
provenance/                 Git, Unity, build log, and file inventory evidence
CHECKSUMS.sha256             SHA-256 for every archive file except itself
MANIFEST.txt                 Sorted file inventory
```

Do not move individual files out of `runtime/Linux`; the executable, `_Data`
directory, and adjacent Unity libraries are one deployment unit.

## Server prerequisites

- Linux x86_64 supported by Unity 6.3.
- NVIDIA T4/T6-class GPU with a compatible production driver.
- Working Vulkan loader and NVIDIA Vulkan ICD.
- Sufficient write permission for the selected output directory.

Verify before testing:

```bash
nvidia-smi -L
sha256sum -c CHECKSUMS.sha256
chmod +x runtime/Linux/V10Stage12GpuPlayer scripts/*.sh
```

Do not launch the GPU player with `-nographics`. The helper uses
`-force-vulkan`, `-batchmode`, and Sentis `GPUCompute`.

## Raw input contract

Raw CSV and CSV.GZ files require `x,y,z`. `dist_center_ft` is optional. Labels
and ground truth are neither required nor read. Choose exactly one mode:

1. `INPUT_CSV`, plus `GROUP_ID` and `SLICE_SEQ` for one arriving slice.
2. `INPUT_DIRECTORY`, plus `GROUP_ID`, for a static directory of slices.
3. `INPUT_MANIFEST` for explicit multi-session or production ordering.

Directory mode extracts the last integer in each filename as `slice_seq`. Use
a manifest whenever filenames cannot produce unique sequences.

## One-slice real-time test

```bash
export INPUT_CSV=/absolute/path/to/unlabeled_slice.csv
export GROUP_ID='production/session0'
export SLICE_SEQ=100
export EXPECTED_SESSIONS=1
unset INPUT_MANIFEST INPUT_DIRECTORY
bash scripts/launch_v10_unity_realtime_handoff.sh
```

## Manifest or session test

```bash
export INPUT_MANIFEST=/absolute/path/to/stage1_manifest.csv
export EXPECTED_SESSIONS=1
unset INPUT_CSV INPUT_DIRECTORY GROUP_ID SLICE_SEQ
bash scripts/launch_v10_unity_realtime_handoff.sh
```

The command-line player processes the supplied snapshot and exits. A
long-lived production Unity service should keep the scene resident and invoke
the same inference manager and native reconstructor as each slice arrives;
restarting the player once per slice is supported for acceptance testing but
adds process-start overhead.

## Monitor

```bash
bash scripts/monitor_v10_unity_realtime_handoff.sh
```

States are `STARTING`, `ACTIVE`, `STALLED`, `COMPLETE`, `FAILED`, and
`EXITED_INCOMPLETE`. Exit code 10 means active, 0 means complete, and 1 means
stalled or failed. `STALLED` means the player exists but its 15-second
heartbeat is older than 300 seconds.

## Output contract

The default output preserves the experiment hierarchy:

```text
output/poleline_voxel_run_session_groups/v4_stage23_quality/unity_native_v10_<stamp>/
  stage1/<sid>/stage1_scores/<relative-parent>/<stem>_stage1.csv
  stage2/<sid>/stage2_objects/<relative-parent>/<stem>_*.csv
  status/
  timing/STAGE12_TIMING_PER_SLICE.csv
  STAGE12_TIMING_SESSION_AVERAGES.txt
  RUN_HEARTBEAT.txt
  STAGE12_COMPLETE.txt or STAGE12_FAILED.txt
```

Set `OUTPUT_BASE` or the complete `RUN_ROOT` before launch when the server uses
a mounted production output volume. Internal Stage 1 and Stage 2 structure is
unchanged.

## Acceptance rule

For release qualification, set `REFERENCE_ROOT` to the accepted opt1-fix2
output and require all of the following:

- `STAGE12_COMPLETE.txt` exists.
- `STAGE12_FAILED.txt` and `FATAL_ERROR.txt` do not exist.
- no `*.failed` status files exist.
- `UNITY_NATIVE_EQUIVALENCE.txt` contains `status=PASS`.
- timing and output counts are complete for every requested slice.

Visual inspection can supplement but never override these gates.
