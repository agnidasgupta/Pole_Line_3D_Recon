# Nebius V4 production-test runbook

This runbook intentionally avoids host Python. Every Python command is executed inside Docker.

## A. Put the supplied test bundle on Nebius

Upload the ZIP to your Nebius VM (for example to your home directory) using `scp` or the Nebius upload mechanism. On Nebius, locate and extract it without assuming a fixed workspace path:

```bash
ARCHIVE=$(find "$HOME" /workspace /tmp -maxdepth 3 -type f -name 'Pole_Line_3D_Recon_V4_Production_Ready_Nebius_Test.zip' -print 2>/dev/null | head -1)
test -n "$ARCHIVE" || { echo 'V4 test archive not found'; exit 1; }
TEST_ROOT="$HOME/v4_production_test_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$TEST_ROOT"
unzip -q "$ARCHIVE" -d "$TEST_ROOT"
V4_DIR=$(find "$TEST_ROOT" -type f -name run_v4_production_tests_on_nebius.sh -printf '%h\n' | head -1)
test -n "$V4_DIR" || { echo 'V4 code directory not found after extraction'; exit 1; }
cd "$V4_DIR"
chmod +x *.sh
printf 'V4_DIR=%s\n' "$V4_DIR"
```

No dataset/session/output directory is entered manually. The V4 launchers discover and persist them.

## B. Build the Docker image

```bash
./build_v4_realtime_image_on_nebius.sh
```

The command starts a background build and prints its persistent log path. After reconnecting, check it with:

```bash
./show_v4_build_state_on_nebius.sh
```

Continue only when it prints `BUILD_RESULT=PASS`. Expected image name unless overridden:

```text
va-v4-realtime:torch241-cu121
```

The build script validates imports and CUDA from **inside the image**.

## C. Run preflight

```bash
./run_v4_realtime_preflight_on_nebius.sh
```

This also starts detached. Check it after reconnecting with:

```bash
./show_v4_production_state_on_nebius.sh
```

Continue only when the latest preflight directory contains `PREFLIGHT_OK.txt`. It performs a host-shell-only discovery smoke test, then inside Docker:

- compiles all Python sources;
- imports all V4 library modules;
- parses every production/training/diagnostic CLI;
- validates source-stage isolation and the 9-gap/450-ft contract;
- runs Stage 1/2/3 synthetic runtime smokes;
- tests missing paths, missing manifests, invalid records, duplicate slice sources, legal slice gaps, malformed manifests, moved artifacts, and invalid span arguments;
- confirms CUDA and dependency versions.

The persistent preflight log/marker contains:

```text
V4_REALTIME_NEBIUS_PREFLIGHT_OK
```

Do not continue on failure. The script prints the persistent `run_context.env` and backup location so the exact environment can be recovered later.

## D. Start the full production acceptance suite

```bash
./run_v4_production_tests_on_nebius.sh
```

This starts a **detached Docker container** and returns immediately. It records the container name and persistent test root. You do not need to keep the SSH connection alive.

The detached suite performs, in order:

1. all code/synthetic checks again inside Docker;
2. H100 `full_cpu` reference versus `active_cpu`, `full_gpu`, and `active_gpu`;
3. Stage 1 score/label equivalence **and Stage 2 topology/geometry/refiner-probability equivalence**;
4. automatic runtime selection;
5. batch-size equivalence + latency sweep;
6. independent-stage test: Stage 1 is saved and moved, Stage 2 consumes the moved Stage 1 boundary, Stage 2 is saved and moved, and Stage 3 is regenerated twice from the moved Stage 2 boundary;
7. rolling quick replay;
8. full auto-selected session replay;
9. strict replay verification and P50/P95 timing summaries;
10. production acceptance marker written only if everything passes.

The test uses the physical Stage 3 contract:

```text
max_sequence_gap = 9
slice_length_ft = 50
max_span_length_ft = 450
max_observed_slice_centers = 10
```

## E. Reconnect and check status

Return to the extracted code directory programmatically if needed:

```bash
V4_DIR=$(find "$HOME" -type f -name run_v4_production_tests_on_nebius.sh -printf '%h\n' 2>/dev/null | sort | tail -1)
cd "$V4_DIR"
./show_v4_production_state_on_nebius.sh
```

If your test tree is under `/workspace` instead of `$HOME`, use:

```bash
V4_DIR=$(find "$HOME" /workspace -type f -name run_v4_production_tests_on_nebius.sh -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
cd "$V4_DIR"
./show_v4_production_state_on_nebius.sh
```

Success is explicit:

```text
V4_PRODUCTION_ACCEPTANCE_OK
```

If it fails, do not rerun blindly. The test root contains `acceptance.log` and `FAILED.txt`, and the run root contains persistent failure/error artifacts.

## F. Optional: prove the stages independently after the gate

The full acceptance suite already performs this test. You can also run the stages individually against the automatically selected session and deployment:

```bash
./run_v4_stage1_on_nebius.sh
./show_v4_production_state_on_nebius.sh
```

After Stage 1 completes:

```bash
./run_v4_stage2_on_nebius.sh
./show_v4_production_state_on_nebius.sh
```

After Stage 2 completes:

```bash
./run_v4_stage3_on_nebius.sh
./show_v4_production_state_on_nebius.sh
```

All three are detached and use durable upstream artifacts. Stage 2 does not require Stage 1 inference to be rerun; Stage 3 does not require Stage 1 or Stage 2 to be rerun.

## G. Optional: run accepted production replay

Only after the exact code/model/calibration/Stage-2 bundle fingerprint has a current acceptance marker:

```bash
./run_v4_production_on_nebius.sh
```

The launcher refuses to run if code or any deployment artifact changed after acceptance.

## H. Build the compact review package

After acceptance (or after an accepted production replay):

```bash
./package_v4_review_bundle_on_nebius.sh
```

Before packaging, verification, timing summarization, and diagnostics collection are run inside Docker. The archive includes the useful review data without copying every large Stage 1 NPZ:

- `run_context.env` and session inventory;
- deployment/code/model/calibration/Stage-2 hashes;
- runtime equivalence gate;
- selected runtime and batch size;
- batch sweep results;
- acceptance log and failure marker if present;
- independent-stage completion markers;
- `realtime_slice_timing.csv`;
- `realtime_timing_summary.{txt,json}`;
- `REALTIME_REPLAY_VERIFICATION.json`;
- Stage 1/2/inference manifests;
- compact diagnostics JSON;
- error artifacts;
- latest Stage 3 reconstruction snapshot;
- archive SHA256.

The script prints the persistent archive path and also writes `LATEST_REVIEW_ARCHIVE.txt`.

## I. Download diagnostics to the Mac

From the local Mac clone/extracted V4 code directory, make the helper executable once:

```bash
chmod +x download_v4_review_bundle_to_mac.sh
```

If your SSH alias is `nebius-va`:

```bash
./download_v4_review_bundle_to_mac.sh
```

If you use a different SSH alias/host:

```bash
REMOTE_HOST='<your-nebius-ssh-alias-or-host>' ./download_v4_review_bundle_to_mac.sh
```

The helper auto-discovers the newest remote review archive, downloads it to:

```text
~/Downloads/v4_production_review/
```

and verifies its SHA256 using macOS `shasum`. Upload the resulting `v4_review_*.tar.gz` file here.

## J. What to send back on failure

Even if acceptance fails, run:

```bash
./show_v4_production_state_on_nebius.sh
```

Then try:

```bash
./package_v4_review_bundle_on_nebius.sh
```

If the replay was not far enough along to package, copy the paths printed by `show_v4_production_state_on_nebius.sh` and download the indicated `acceptance.log`, `FAILED.txt`, `run_context.env`, runtime-gate JSON, or error JSON with `scp`. Do not delete the persistent V4 production directory between attempts.
