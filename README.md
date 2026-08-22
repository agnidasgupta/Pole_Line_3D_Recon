# Pole_Line_3D_Recon

Three-stage 3D utility-pole and powerline detection/reconstruction pipeline for voxelized LiDAR CSV slices.

The current codebase is the consolidated V6.2 **teacher-recall** pipeline. It includes the latest runtime-hardening, resumable/performance Stage-3 reconstruction, and one-session timing utilities.

## Architecture

### Stage 1 - local voxel detection

- Predicts background, pole, and powerline evidence from each slice.
- Uses V4 as a frozen **positive-only powerline teacher** during training to preserve V4's useful line recall without copying its background decisions.
- Uses semantic/classification, focal/asymmetric, soft-IoU, objectness, and local geometry/continuity terms.
- Training and Stage-1 inference remain slice-local; world/slice-center coordinates are not used here.

### Stage 2 - local object extraction and refinement

- Converts Stage-1 probabilities into pole candidates and conductor fragments.
- Uses strong/weak hysteresis for line support so weak but connected conductor voxels can survive.
- Uses connected components and local geometric descriptors (orientation, extent, verticality, tortuosity, etc.).
- Applies learned local pole/line refiners.
- Plausible GT disagreements can be excluded as ambiguous rather than forced into negative training.

### Stage 3 - session/world reconstruction

- Converts local objects to world coordinates using slice metadata.
- Merges duplicate poles and joins compatible conductor fragments across slices.
- Uses direction, lateral offset, Z continuity, gap, span-length, and slice-range constraints.
- Completes strongly supported fragmented pole-to-pole spans.
- Infers a hidden pole only from multiple independently supported, nonparallel approaching spans under strict geometry/support rules.
- Prevents self-intersections and polygon/cycle-forming synthetic topology while allowing parallel conductors.
- Uses bounded pole-height adjustment.
- Uses spatial indexing (`scipy.spatial.cKDTree`) and supports per-session Stage-3 resume.

## Repository contents

Core training/inference:

- `train_v6_stage1.py`
- `full_scene_evaluate_v6_stage1.py`
- `prepare_v62_dataset.py`
- `select_v6_candidate.py`
- `mine_v62_local_components.py`
- `train_v62_local_refiners.py`
- `infer_v62_stage1_stage2.py`
- `v6_common.py`, `v6_components.py`, `v6_predict.py`, `v62_local.py`
- `precision_common.py`, `voxel_common.py`

Reconstruction:

- `reconstruct_v62_stage3.py`
- `run_v62_stage3.sh`
- `inspect_v62_stage3_resume_state.sh`

Primary launchers:

- `run_v62_teacher_training_on_nebius.sh` - training only
- `run_v62_teacher_inference_on_nebius.sh` - inference only
- `run_v62_teacher_reconstruction_on_nebius.sh` - reconstruction only
- `run_v62_teacher_all_on_nebius.sh` - all stages sequentially; completed stages are skipped

Timing utilities:

- `time_v62_one_session_inference.py`
- `time_v62_one_session_reconstruction.py`
- `run_v62_one_session_inference_timed_on_nebius.sh`
- `run_v62_one_session_reconstruction_timed_on_nebius.sh`
- `run_v62_one_session_timing_all_on_nebius.sh`

Diagnostics:

- `collect_v62_diagnostics.py`
- `compare_v4_v62_line_metrics.py`
- `check_v62_teacher_status.sh`

Additional design/change notes are kept in the `README_*.md` files and `V4_V62_LINE_RECALL_ANALYSIS.md`.

## External artifacts intentionally NOT stored in Git

The repository does not contain source data, NPZ caches, trained models, or generated outputs. In particular, V6.2 training expects the V4 teacher artifacts to exist outside the repository, by default under:

```text
/outputs/poleline_voxel_run_session_groups/precision_v4/train/precision_best.pt
/outputs/poleline_voxel_run_session_groups/precision_v4/full_val/calibration.json
```

Raw data defaults to:

```text
/data/voxel_csv_combined
```

Host outputs default to:

```text
/workspace/voxel_poleline/outputs
```

All of these paths can be overridden with environment variables.

## Build the Docker image

If the existing image is not already available:

```bash
docker build -t va-voxel-poleline:v6.2-three-stage .
```

The Nebius launchers use Docker with GPU access for training/inference and bind-mount this repository read-only into the container.

## Run training only

```bash
./run_v62_teacher_training_on_nebius.sh
sudo docker logs -f poleline-v62-teacher-training
```

## Run inference only

```bash
RESUME=1 ./run_v62_teacher_inference_on_nebius.sh
sudo docker logs -f poleline-v62-teacher-inference
```

## Run reconstruction only

Fresh Stage 3:

```bash
./run_v62_teacher_reconstruction_on_nebius.sh
```

Resume a partially completed Stage 3:

```bash
RESUME_STAGE3=1 ./run_v62_teacher_reconstruction_on_nebius.sh
```

Monitor:

```bash
sudo docker logs -f poleline-v62-teacher-reconstruction
```

## Run all stages

```bash
./run_v62_teacher_all_on_nebius.sh
```

Completed training/inference/reconstruction stages are skipped by completion markers. To allow a partial Stage 3 to resume when using the all-stage launcher:

```bash
RESUME_STAGE3=1 ./run_v62_teacher_all_on_nebius.sh
```

## One-session timing

```bash
SESSION_FILTER="59768101-C4990BB-2026/session3" \
MODE=both \
./run_v62_one_session_timing_all_on_nebius.sh
```

This reports inference timing per slice and rolling Stage-3 reconstruction timing per slice for the same session.

## Download experiment outputs to a Mac

The helper excludes model/checkpoint files and NPZ caches:

```bash
./download_v62_teacher_results_to_mac.sh
```

Override destinations as needed:

```bash
REMOTE_HOST=nebius-va \
LOCAL_RUN="$HOME/Documents/VEG_Data/POLE_Voxel/v62_teacher_recall_complete/" \
./download_v62_teacher_results_to_mac.sh
```

## Git hygiene

`.gitignore` excludes datasets, generated outputs, NPZ files, model/checkpoint files, local environments, logs, archives, and common secret-key file patterns. Before every push, still inspect staged files with:

```bash
git status
git diff --cached --stat
git diff --cached
```
