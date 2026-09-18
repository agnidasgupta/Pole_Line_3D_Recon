# V6.2 teacher-recall pipeline

This revision keeps the V6.2 three-stage architecture while directly targeting the observed V4-vs-V6.2 powerline recall gap.

## Stage 1: V4 positive-teacher distillation

The V4 `precision_best.pt` checkpoint is used in two ways:

1. it initializes the V6.2 fine/coarse student network as before; and
2. it remains frozen during training as a **positive-only powerline teacher**.

For a GT-background voxel, the teacher can add weak positive line supervision only when V4 line confidence is high, V4 line confidence exceeds pole confidence, and local occupancy is not strongly vertical. No V4 negative/background teacher loss is used, so V4 false positives are not copied as negative decisions.

Teacher loss supervises the V6.2 line head, semantic line class, and objectness head. Default teacher weight is `0.30`.

Local conductor physics is also less axis-biased. Horizontal support is evaluated over X, Y, +45-degree, and -45-degree XY directions, with small Z tolerance. This reduces suppression of diagonal conductors.

Validation/model selection uses separate pole and line targets. Powerline calibration is deliberately recall-aware because strict GT is known to omit/misalign some real conductors.

Default Stage-1 recall settings:

- `LAMBDA_V4_LINE_TEACHER=0.30`
- `V4_TEACHER_MIN_LINE_SCORE=0.30`
- `V4_TEACHER_LINE_OVER_POLE_MARGIN=0.05`
- `STAGE1_LINE_TARGET_PRECISION=0.58`
- `STAGE1_LINE_TARGET_RECALL=0.88`
- `STAGE1_LINE_TARGET_IOU=0.52`
- `STAGE1_LINE_RECALL_WEIGHT=2.5`

## Stage 2: softer line gating

Powerline candidates use strong/weak hysteresis rather than a single hard threshold:

- strong seed: `LINE_CANDIDATE_THRESHOLD=0.08`
- connected weak support: `LINE_WEAK_THRESHOLD=0.04`
- line/pole competition ratio: `0.55`
- minimum candidate component: 3 voxels

Weak voxels survive only when they belong to a connected component containing a strong line seed. This can preserve an occluded/sparse wire segment without turning isolated low-score voxels into line components.

The old strict physical acceptance rule is replaced by a loose impossible-geometry gate. The Stage-2 learned refiner therefore gets to evaluate short, sparse, slice-edge, and occluded candidates. Plausible GT disagreements remain ambiguous and are excluded from refiner-negative training.

The Stage-2 line refiner is selected with a recall-oriented target (`recall=0.98`, strict-GT precision floor `0.60` by default).

## Stage 3: circuit-complete topology-safe reconstruction

The package includes the latest reconstruction logic:

- bounded pole-height adjustment; conductor attachments cannot create extremely tall poles;
- completion of geometrically supported fragmented conductor tracks between two poles;
- hidden-pole inference only when multiple independently pole-anchored, nonparallel spans converge on a supported location;
- missing/discarded slice gaps do not by themselves create hidden poles;
- no self-closing conductor tracks or synthetic polygon/triangle completion;
- black poles and cyan conductors.

## Independent execution

The three operational phases are independently restartable on Nebius:

- training + held-out evaluation + Stage-2 refiner training: `./run_v62_teacher_training_on_nebius.sh`
- all-data Stage-1+2 inference using existing trained artifacts: `./run_v62_teacher_inference_on_nebius.sh`
- Stage-3 reconstruction using existing Stage-2 objects: `./run_v62_teacher_reconstruction_on_nebius.sh`

Or run all phases sequentially with `./run_v62_teacher_all_on_nebius.sh`.

All output is stored under:

`/workspace/voxel_poleline/outputs/poleline_voxel_run_session_groups/v62_teacher_recall`

The Nebius wrappers bind-mount this source directory into the already available `va-voxel-poleline:v6.2-three-stage` image, so a Docker rebuild is not required.

## Result artifacts

Training produces history CSVs, loss/metric plots, V4-teacher diagnostics, threshold searches, full-scene validation/test metrics and confusion matrices, Stage-2 PR/feature-importance plots, component target counts, logs, and a V4-vs-V6.2 strict-GT comparison plot/CSV/JSON. Model files remain on Nebius for inference.

Inference produces row-level gzip CSVs, Stage-2 pole/line/vertex CSVs, inference manifests, strict-GT metrics where labels exist, confusion-matrix PNGs, logs, and progress/completion JSON. Internal Stage-2 point NPZs remain on Nebius for restart/debug support.

Reconstruction produces aggregate and per-session pole/conductor/span CSVs, 2D/3D PNGs, join/span-completion/hidden-pole/topology audit CSVs, logs, summaries, and `COMPLETED.json`.

## Mac download policy

Run `download_v62_teacher_results_to_mac.sh` on the Mac after completion. It recursively downloads logs, JSON/TXT metrics, CSV/CSV.GZ outputs, PNG/JPG diagnostics, inference CSVs, and reconstruction products while excluding:

- `*.npz`
- `*.pt`, `*.pth`, `*.ckpt`
- `*.joblib`, `*.pkl`
- `*.onnx`, `*.engine`
- `__pycache__/`

This means models and caches stay on Nebius, but the analysis/results package is copied locally.
