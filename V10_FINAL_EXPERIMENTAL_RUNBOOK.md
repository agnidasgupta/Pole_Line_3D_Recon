# V10 final experimental branch runbook

Branch: `v4-stage2-stage1-electrical-v10`

The accepted Stage-2 behavior is frozen. The final optimization is `opt1-fix2`:
dead quadratic bridge enumeration is removed and pole-contact lookup uses a voxel
hash. The all-session gate requires 30/30 sessions, 30 voxel-support reports, and
30 baseline-equivalence reports before writing completion markers.

## 1. Install the source release on the Mac branch

From Terminal on the Mac (commands contain only ASCII characters):

```bash
REPO=/path/to/Pole_Line_3D_Recon
ARCHIVE=/Users/agni/Downloads/v10_final_experimental_source_20260911.tar.gz
cd "$REPO"
git switch v4-stage2-stage1-electrical-v10
git status --short
tar -xzf "$ARCHIVE" -C "$REPO"
git status --short
```

Review the changed inventory, then commit and push from the Mac, where GitHub
authentication already works:

```bash
git add v4 ops V10_FINAL_EXPERIMENTAL_RUNBOOK.md
git commit -m "Finalize quality-equivalent V10 Stage2 and Unity Sentis export"
git push -u origin v4-stage2-stage1-electrical-v10
```

Do not merge the branch to the production/default branch until the Unity gates pass.

## 2. Upload the source archive to Nebius

```bash
scp "$ARCHIVE" nebius-va:/home/agni/
ssh nebius-va
```

On Nebius:

```bash
REPO=/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10
tar -xzf /home/agni/v10_final_experimental_source_20260911.tar.gz -C "$REPO"
find "$REPO/v4" "$REPO/ops" -type f -name '*.sh' -exec chmod +x {} +
```

## 3. Optional training entry points

The accepted Stage-1 checkpoint and Stage-2 refiner must remain unchanged for the
inference/export experiment. The following commands are for a separate retraining
experiment only.

Stage 1:

```bash
bash "$REPO/v4/run_v4_stage1_training_on_nebius.sh"
```

Stage 2 refiner training from the existing Stage-1 checkpoint:

```bash
bash "$REPO/v4/run_v4_stage2_training_on_nebius.sh"
```

Never overwrite `precision_v4` or `v4_realtime/stage2_refiner`; use the timestamped
experimental defaults or explicit new output paths.

## 4. Run the quality-equivalent optimized Stage 2 experiment

The baseline pointer must name the visually accepted complete V10 run. Confirm it:

```bash
cat /home/agni/LATEST_V10_FAULT_TOLERANT_STAGE2_RUN.txt
```

Launch:

```bash
bash "$REPO/ops/v4_stage2_stage1_electrical_v10_opt/launch_v10_stage2_opt_experiment.sh"
```

Monitor at any time:

```bash
bash "$REPO/ops/v4_stage2_stage1_electrical_v10_opt/monitor_v10_stage2_opt_experiment.sh"
```

Interpretation:

- `STATE=RUNNING` plus output age below 900 seconds: active or normally waiting.
- `HEALTH=CHECK_POSSIBLE_STALL`: inspect the current session log and `docker ps`.
- `STATE=EXITED` plus `READY_TO_PACKAGE=YES`: complete.
- Any failed/equivalence count or missing completion marker: do not package as a
  successful result; restart with `RESUME=1` after correcting the reported error.

Package only after `READY_TO_PACKAGE=YES`:

```bash
bash "$REPO/ops/v4_stage2_stage1_electrical_v10_opt/package_v10_stage2_opt_experiment.sh"
```

Download on the Mac while preserving the run-root directory:

```bash
NEBIUS=nebius-va
ARCHIVE=$(ssh "$NEBIUS" 'cat /home/agni/LATEST_V10_STAGE2_OPT_ARCHIVE.txt')
scp "$NEBIUS:$ARCHIVE" "$NEBIUS:$ARCHIVE.sha256" /Users/agni/Downloads/
cd /Users/agni/Downloads
shasum -a 256 -c "$(basename "$ARCHIVE").sha256"
mkdir -p v10_stage2_results
tar -xzf "$(basename "$ARCHIVE")" -C v10_stage2_results
```

The archive contains the unchanged hierarchy
`fault_tolerant_v10_<stamp>/stage2/<session-safe-id>/...`, plus timing, status,
logs, manifests, inventories, and completion markers.

## 5. Generate the Unity/Sentis package on Nebius

This uses the existing accepted checkpoint, calibration, Stage-2 joblib bundle,
and Velasco profile source. It creates FP32 and FP16 ONNX, verifies them with real
patches, benchmarks available ONNX Runtime providers including TensorRT, exports
the Stage-2 tree provenance JSON, and copies the C# manager.

```bash
bash "$REPO/ops/v4_stage12_unity_sentis/launch_v10_unity_export_on_nebius.sh"
```

Monitor:

```bash
bash "$REPO/ops/v4_stage12_unity_sentis/monitor_v10_unity_export_on_nebius.sh"
```

`READY_TO_PACKAGE` means the export completed. `POSSIBLY_STALLED` means no log or
artifact changed for 15 minutes while the launcher PID still exists; inspect the
shown process/container table and log before stopping anything.

Package:

```bash
bash "$REPO/ops/v4_stage12_unity_sentis/package_v10_unity_export_on_nebius.sh"
```

Download on the Mac:

```bash
NEBIUS=nebius-va
ARCHIVE=$(ssh "$NEBIUS" 'cat /home/agni/LATEST_V10_UNITY_EXPORT_ARCHIVE.txt')
scp "$NEBIUS:$ARCHIVE" "$NEBIUS:$ARCHIVE.sha256" /Users/agni/Downloads/
cd /Users/agni/Downloads
shasum -a 256 -c "$(basename "$ARCHIVE").sha256"
mkdir -p v10_unity_export
tar -xzf "$(basename "$ARCHIVE")" -C v10_unity_export
```

Use FP32 first. FP16 is a candidate only when both the sidecar field
`onnx.fp16_deployable` and the later Unity-vs-Python parity gate are true.

## 6. Timing records

The all-session optimized run writes:

- `timing/stage2/<session-safe-id>.csv`: one row per slice.
- `STAGE2_TIMING_SESSION_AVERAGES.txt`: one average block per session and a
  slice-weighted total.
- `status/<session-safe-id>.quality_equivalence.json`: output-quality gate.

The accepted reference timing report is also committed at
`ops/v4_stage2_stage1_electrical_v10_opt/STAGE2_TIMING_SESSION_AVERAGES_ACCEPTED.txt`.
