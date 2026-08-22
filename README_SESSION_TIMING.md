# V6.2 one-session inference and Stage-3 timing

This add-on does not retrain anything and does not modify the trained model.

## Inference timing

`time_v62_one_session_inference.py` filters the raw source tree to one exact group id such as:

`59768101-C4990BB-2026/session3`

It launches the existing runtime-hardened `infer_v62_stage1_stage2.py` once, so the Stage-1 model is loaded only once. It records model-load/compile time separately and records per-slice wall time after the model-ready message. Per-slice timing therefore includes CSV read, Stage-1 prediction, Stage-2 component extraction/refiner inference, CSV/NPZ writes, and bookkeeping.

Outputs include:
- `slice_inference_timing.csv`
- `session_inference_timing.json`
- `timed_inference.log`
- the normal inference CSVs and Stage-2 object files for that session.

## Reconstruction timing

Stage 3 is inherently multi-slice; a complete span cannot be reconstructed from one slice in isolation. Two measurements are therefore produced:

1. `batch`: reconstruct the complete selected session once and report total wall time and an amortized equivalent seconds/slice.
2. `rolling`: for every observed slice, run Stage 3 with `--latest_slice`. The window is the latest slice plus observed slices whose numeric sequence lies within the previous `MAX_SPAN_SLICES` indices (default 9). This is the actual per-slice rolling/realtime latency measurement.

Outputs include:
- `slice_reconstruction_timing.csv` (rolling actual per-slice latency)
- `session_reconstruction_timing.json`
- `batch_reconstruction.log`
- `rolling_logs/slice_<seq>.log`
- batch reconstruction outputs.

Rolling reconstruction directories are deleted after timing by default to avoid creating a large duplicate result tree. Set `KEEP_ROLLING_OUTPUTS=1` to keep them.
