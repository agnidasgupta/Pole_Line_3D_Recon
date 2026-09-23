# V10 Stage 1 + Stage 2 Unity/Sentis export

This directory packages the accepted V4 Stage-1 neural inference model and the
accepted V10 voxel-supported Stage-2 reconstruction for Unity Server 6.3 LTS.

## Quality boundary

- `v10_stage1_voxelnet3d_fp32.onnx` is the deployment default.
- FP16 is only enabled by the C# manager when `onnx.fp16_deployable` is `true` in
  the generated sidecar. The exporter sets that flag only after zero calibrated
  label changes on real dataset patches.
- Stage 2 is **not** converted into ONNX. It contains connected-component,
  shortest-path, pole-refiner, and geometric audit code rather than a tensor
  network. The exact accepted Python code is retained and runs in Docker from the
  Unity Stage-1 CSV handoff.
- TensorRT is used only for Nebius ONNX compatibility and latency benchmarking.
  A TensorRT `.plan` is not a Unity Sentis model and is not copied into the Unity
  deployment unless explicitly requested as a benchmark artifact.

## Contents after Nebius export

```text
unity_package/
  Assets/
    Scripts/
      V10Stage12SentisInferenceManager.cs
    Editor/
      V10SentisFp16Quantizer.cs
    StreamingAssets/V10Stage12/
      v10_stage1_voxelnet3d_fp32.onnx
      v10_stage1_voxelnet3d_fp16.onnx
      v10_stage12_sidecar.json
      v10_stage2_refiner_trees.json
      UNITY_READINESS.txt
  run_unity_stage1_csv_stage2.py
  README.md
  SHA256SUMS.txt
```

`v10_stage2_refiner_trees.json` is an inspectable export of the scikit-learn
ExtraTrees bundle. It is supplied for provenance and future native-C# parity work;
the accepted runtime continues to load the original joblib bundle in Docker.

## Unity use

1. Install `com.unity.ai.inference` version `2.6.1`.
2. Copy the exported `Assets` directory into the Unity project.
3. Import the FP32 ONNX model and assign it plus `v10_stage12_sidecar.json` to
   `V10Stage12SentisInferenceManager`.
4. Keep `requestFp16` disabled for the first parity run.
   The optional Editor menu utility performs Sentis-native FP16 weight
   quantization; this candidate also requires the same parity gate.
5. Do not launch a headless server with a configuration that disables graphics
   devices; `BackendType.GPUCompute` requires compute-shader support. Use the CPU
   backend only as a verified fallback.
6. Call `InferSliceAsync`, then `WriteStage1Csv`.
7. Mount the CSV, repository, model bundle, calibration, profile, and output root
   into the repository Docker image and run `run_unity_stage1_csv_stage2.py`.

Example container command (paths are examples and must be mounted):

```bash
docker run --rm --mount type=bind,source=/srv/v10,target=/exchange \
  --mount type=bind,source=/workspace/voxel_poleline/outputs,target=/outputs \
  --mount type=bind,source="$PWD/v4",target=/workspace/v4,readonly \
  --mount type=bind,source="$PWD/ops/v4_stage2_stage1_electrical_v10_opt",target=/workspace/quality,readonly \
  --mount type=bind,source="$PWD/ops/v4_stage12_unity_sentis",target=/workspace/unity,readonly \
  --workdir /workspace/v4 \
  -e PYTHONPATH=/workspace/v4:/workspace/quality \
  va-v4-realtime:torch241-cu121 \
  python /workspace/unity/run_unity_stage1_csv_stage2.py \
    --stage1-csv /exchange/stage1.csv \
    --output-dir /exchange/stage2/VELASCO_CUT_CP__session1 \
    --relative-path VELASCO_CUT_CP/session1/slice.csv \
    --file-id slice --slice-seq 0 --group-id VELASCO_CUT_CP/session1 \
    --stage2-bundle /outputs/poleline_voxel_run_session_groups/v4_realtime/stage2_refiner/local_refiner_bundle.joblib \
    --calibration-json /outputs/poleline_voxel_run_session_groups/precision_v4/full_val/calibration.json \
    --profile-json /exchange/selected_electrical_profile.json
```

The production service should use an argument list rather than shell-concatenated
user input when spawning this process.

## Required release gates

The Unity candidate is not promoted merely because ONNX Runtime passes. Run these
gates on representative and difficult slices, especially
`VELASCO_CUT_CP/session1` ordinals 20–39:

1. FP32 Sentis Stage-1 labels match the Python checkpoint labels at every occupied
   voxel.
2. FP32 scores remain within the recorded tolerance and do not cross either
   calibrated threshold.
3. The Docker Stage-2 outputs pass the existing voxel-support validator.
4. Line/electrical CSVs are byte-identical to the accepted Stage-2 reference;
   production pole/component values pass the existing `1e-9` semantic tolerance.
5. FP16 may be enabled only if it independently passes the same decision and
   Stage-2 equivalence gates.

## Why no additional Stage-2 optimization is merged

The accepted optimization already removes the dead quadratic disconnected-bridge
enumeration and replaces repeated pole-contact scanning with a coordinate hash. On
3,738 slices it reduced weighted Stage-2 compute from 2,065.755 ms to 233.851 ms
per slice (8.83x). The remaining production refiner/parameterization and electrical
geometry blocks directly determine accepted output. A prior shortcut changed
production outputs, so further changes remain separate experiments until they pass
all gates above.
