# V10 Unity-native Stage 1 + Stage 2 experiment

This directory removes the Python handoff from the earlier Unity package.

- **Stage 1:** the unchanged accepted FP32 ONNX model runs through Unity Inference Engine/Sentis 2.6.1.
- **Stage 2 poles:** native C# reproduces sparse 26-connected components, local features, the exported ExtraTrees refiner, physical gates, and pole parameterization.
- **Stage 2 conductors:** native C# reproduces the accepted `opt1-fix2` graph paths. Every Stage 1 class-2 voxel is assigned exactly once, disconnected components are never bridged, every simplified chord remains inside Stage 1 line-voxel cells, sharp turns split, open ends remain open, and pole attachment requires direct inferred pole/line contact.
- **Runtime dependencies:** Unity player only. No Python, NumPy, pandas, sklearn, joblib, Docker, Nebius service, or network is used by the built player.

Stage 2 is intentionally **not ONNX**: it is a variable-size graph/connected-component/tree pipeline rather than a neural tensor graph. It runs in native C# in the same Unity process immediately after Sentis inference.

## Corrected defects

The package corrects the earlier hybrid export by:

1. replacing `run_unity_stage1_csv_stage2.py` with native C# Stage 2;
2. serializing the accepted ExtraTrees models to a compact, SHA-256-bound binary asset;
3. matching Stage 1 valid-row filtering, NumPy round-to-even coordinates, duplicate-last behavior, distance normalization order, z-stable row order, 64-cube patches, and 48-cube active cores;
4. serializing Sentis worker warmup and inference with one semaphore;
5. preserving production pole component ID order and deterministic pole-contact order;
6. using recoverable atomic writes, resumable slices, per-slice rejection markers, a 15-second heartbeat, and explicit final markers;
7. applying an automatic reconstruction CSV/electrical equivalence gate against an accepted reference run when `REFERENCE_ROOT` is supplied. The diagnostic `_components.csv` is emitted but is not byte-gated because PCA vector sign is mathematically arbitrary; accepted poles, lines, vertices, voxel identity, tracks, attachments, empty bridge files, and electrical invariants are gated.

FP16 remains blocked because the supplied sidecar records nonzero real-patch label mismatches. The builder always selects FP32. A TensorRT engine is not consumed by Sentis. A dedicated-server build defaults to Sentis CPU because `-nographics` can have no graphics device; an optional standard Linux player build uses Sentis `GPUCompute` and must have a working graphics context.

## Repository branch

Keep this work separate from the accepted Stage 2 branch:

```bash
REPO=/Users/agni/dev/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10
git -C "$REPO" fetch origin
git -C "$REPO" switch -c v4-stage12-unity-native-v10 origin/v4-stage2-stage1-electrical-v10
tar -xzf /Users/agni/Downloads/v10_unity_native_stage12_source.tar.gz -C "$REPO"
git -C "$REPO" add ops/v4_stage12_unity_native
git -C "$REPO" commit -m "Add native Unity V10 Stage1 and Stage2 experiment"
git -C "$REPO" push -u origin v4-stage12-unity-native-v10
```

If that branch already exists locally, switch to it and rebase onto its remote instead of creating it again. Never force-push the accepted branch.

## Build assets (Nebius, Docker only)

Upload this source package to Nebius or pull the experimental branch. The completed prior export must contain the two ONNX files, sidecar, and `v10_stage2_refiner_trees.json`.

```bash
cd /workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10
export REPO=$PWD
export EXPORT_ROOT=/workspace/voxel_poleline/outputs/poleline_voxel_run_session_groups/v4_stage23_quality/unity_export_v10_<STAMP>
bash ops/v4_stage12_unity_native/scripts/prepare_v10_unity_model_assets_on_nebius.sh
```

Download the path written to `/home/agni/LATEST_V10_UNITY_NATIVE_MODEL_ASSETS.txt` and its `.sha256` from the Mac using the existing SSH alias `nebius-va`.

## Create the Unity server player on Mac

1. Create/open a Unity 6.3 LTS project and install Inference Engine/Sentis **2.6.1**.
2. Install the source and extract the model archive:

```bash
export UNITY_PROJECT=/Users/agni/dev/V10Stage12Unity
bash /path/to/repo/ops/v4_stage12_unity_native/scripts/install_v10_unity_native_code_on_mac.sh
tar -xzf /Users/agni/Downloads/v10_unity_native_model_assets_<STAMP>.tar.gz -C "$UNITY_PROJECT"
```

3. Install Unity's Linux Dedicated Server build-support module, then build:

```bash
export UNITY_EDITOR=$(find /Applications/Unity/Hub/Editor -path '*/6000.3*/Unity.app/Contents/MacOS/Unity' -type f -print -quit)
bash /path/to/repo/ops/v4_stage12_unity_native/scripts/build_v10_unity_linux_server_on_mac.sh
```

Confirm `"$UNITY_EDITOR" -version` reports Unity 6.3 before building.

The command above makes the reliable headless CPU build. For a GPU experiment, set `V10_UNITY_BUILD_KIND=player` before running it, upload `V10Stage12GpuPlayer`, and set `UNITY_HEADLESS=0` when launching. Qualify CPU and GPU builds independently with the same automatic equivalence gate.

## Input contract

The player accepts either one manifest file or a directory containing per-session `stage1_manifest.csv` files. Required columns are `group_id`, `slice_seq`, `relative_path`, and either `source_csv` or `source`. Each source file may be CSV or CSV.GZ and must contain local `x,y,z`; `dist_center_ft` is optional. See `UNITY_INPUT_MANIFEST_EXAMPLE.csv`.

This is raw Stage 1 input, not precomputed Stage 1 inference. The player creates the Stage 1 result itself and passes it directly to native Stage 2.

## Run as a separate Nebius experiment

Upload the complete `Builds/Linux` directory and `scripts/` directory. Then:

First create and run the definitive Velasco ordinal 20–39 regression. The helper copies only the matching accepted reference files and creates the raw-input manifest automatically:

```bash
export REPO=/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10
export STAGE1_ROOT=/workspace/voxel_poleline/outputs/poleline_voxel_run_session_groups/v4_production/full_dataset_runs/d9977c39c443f5fa14f8/20260825T203403Z/stage1
export ACCEPTED_ROOT=/workspace/voxel_poleline/outputs/poleline_voxel_run_session_groups/v4_stage23_quality/<ACCEPTED_OPT1_FIX2_RUN>
bash ops/v4_stage12_unity_native/scripts/prepare_v10_unity_regression_subset_on_nebius.sh

export UNITY_SERVER=/home/agni/v10_unity_server/V10Stage12Server
export INPUT_MANIFEST=$(cat /home/agni/LATEST_V10_UNITY_REGRESSION_MANIFEST.txt)
export REFERENCE_ROOT=$(cat /home/agni/LATEST_V10_UNITY_REGRESSION_REFERENCE.txt)
export EXPECTED_SESSIONS=1
export RESUME=1
bash "$REPO/ops/v4_stage12_unity_native/scripts/launch_v10_unity_native_experiment.sh"
```

Proceed to the all-session experiment only after the monitor reports `COMPLETE` and `UNITY_NATIVE_EQUIVALENCE.txt` contains `status=PASS`:

```bash
export UNITY_SERVER=/home/agni/v10_unity_server/V10Stage12Server
export INPUT_MANIFEST=/workspace/voxel_poleline/outputs/poleline_voxel_run_session_groups/v4_production/full_dataset_runs/d9977c39c443f5fa14f8/20260825T203403Z/stage1
export EXPECTED_SESSIONS=30
export RESUME=1
export REFERENCE_ROOT=/workspace/voxel_poleline/outputs/poleline_voxel_run_session_groups/v4_stage23_quality/<ACCEPTED_OPT1_FIX2_RUN>
bash "$REPO/ops/v4_stage12_unity_native/scripts/launch_v10_unity_native_experiment.sh"
```

`REFERENCE_ROOT` makes release acceptance automatic. Omit it only for exploratory timing; such a run is not parity-qualified.

Output stays under:

```text
.../v4_stage23_quality/unity_native_v10_<STAMP>/
  stage1/<sid>/stage1_scores/<relative-parent>/<stem>_stage1.csv
  stage2/<sid>/stage2_objects/<relative-parent>/<stem>_*.csv
  status/
  timing/STAGE12_TIMING_PER_SLICE.csv
  STAGE12_TIMING_SESSION_AVERAGES.txt
```

## Monitor without guessing

```bash
bash "$REPO/ops/v4_stage12_unity_native/scripts/monitor_v10_unity_native_experiment.sh"
```

Exit code `10` means active, `0` means complete, and `1` means stalled/failed/exited incomplete. `STALLED` means the PID exists but `RUN_HEARTBEAT.txt` is older than 300 seconds. Do not package unless `STAGE12_COMPLETE.txt` exists and, when a reference was supplied, `UNITY_NATIVE_EQUIVALENCE.txt` says `status=PASS`.

## Package and download

On Nebius:

```bash
bash "$REPO/ops/v4_stage12_unity_native/scripts/package_v10_unity_native_results.sh"
```

On Mac:

```bash
NEBIUS=nebius-va
ARCHIVE=$(ssh "$NEBIUS" 'cat /home/agni/LATEST_V10_UNITY_NATIVE_ARCHIVE.txt')
scp "$NEBIUS:$ARCHIVE" "$NEBIUS:$ARCHIVE.sha256" /Users/agni/Downloads/
cd /Users/agni/Downloads
shasum -a 256 -c "$(basename "$ARCHIVE").sha256"
```

The archive contains the complete run root, so extracting it preserves the same Stage 1/Stage 2 session and relative-path hierarchy.

## Release rule

Do not replace the accepted model with this port merely because it compiles. The automated equivalence report must pass on the definitive Velasco ordinals 20–39 and then on all 30 sessions. A failed report is a failed experiment; no manual CSV correction or visual override is permitted.
