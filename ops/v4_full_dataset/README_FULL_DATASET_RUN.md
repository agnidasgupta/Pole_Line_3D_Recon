# V4 full-dataset production operations

This directory is intentionally **external to the accepted `v4/` directory** so operational tooling can evolve without changing the accepted V4 deployment fingerprint.

Recommended repository location:

```text
Pole_Line_3D_Recon/
├── v4/                     # accepted fingerprinted production implementation
└── ops/
    └── v4_full_dataset/    # this operational add-on
```

## Full-dataset stage execution

Each stage is independently executable from the durable output of the preceding stage:

1. `run_v4_all_stage1_on_nebius.sh` — Stage 1 inference for every valid discovered session.
2. `run_v4_all_stage2_on_nebius.sh` — Stage 2 only, using the latest completed full-dataset Stage 1 artifacts.
3. `run_v4_all_stage3_on_nebius.sh` — Stage 3 only, using the latest completed full-dataset Stage 2 artifacts.
4. `run_v4_all_reconstruction_on_nebius.sh` — convenience wrapper: Stage 2 followed by Stage 3.
5. `run_v4_all_in_one_on_nebius.sh` — convenience wrapper: Stage 1, Stage 2, then Stage 3.

The accepted V4 code is mounted read-only inside Docker. Project Python is never run on the Nebius host.

## Monitoring and recovery

- `show_v4_all_data_state_on_nebius.sh` — monitor the latest persistent run after reconnecting.
- Stage markers (`PHASE1_STAGE1_OK.txt`, `PHASE2_STAGE2_OK.txt`, `PHASE3_STAGE3_OK.txt`) define durable restart boundaries.
- Deployment and input-inventory hashes are checked before a later stage can reuse earlier results.
- Stage 1 `.npz` artifacts remain on Nebius so Stage 2 can be rerun without repeating inference.

## Packaging and download

- `package_v4_all_data_results_on_nebius.sh` packages inference CSV.GZ, Stage 2/3 reconstruction files, metrics, timings, manifests, context, and logs.
- `download_v4_all_data_results_to_mac.sh` runs on the Mac. It first uses the package pointer when available and otherwise discovers the newest result archive directly, then verifies SHA256.
- `make_v4_compact_review_from_results_on_mac.sh` creates a much smaller review archive from the large result package without extracting the per-voxel inference CSV.GZ files.

The package safety gate rejects `.npz`, `.pt`, `.pth`, `.ckpt`, `.joblib`, `.onnx`, `.engine`, and `.safetensors` files.

## Stage 3 production contract

Stage 3 remains constrained to nine 50-ft sequence intervals (450 ft):

```text
newest sequence = S
allowed window  = [S-9, S]
maximum gap     = 9 sequence increments
slice interval  = 50 ft
maximum span    = 450 ft
observed centers = at most 10
```

Missing sequence numbers are allowed. Future slices are prohibited.
