# E8 external-source record

- Update date: 2026-09-24 UTC
- E8 branch: v4-stage1-inference-opt2-e8-pipeline
- Base experiment: origin/v4-stage1-inference-opt2 (accepted E7).
- Imported source: origin/harpreet/v4-h100-opt.
- Imported scope: ops/v4_stage1_inference_opt2 plus v4/v4_input_prefetch.py
  and v4/v4_output_writer.py.
- Enabled E8 flags: PREFETCH_INPUTS=1, PREPARE_CORE_SCHEDULE=1,
  PREFETCH_DEPTH=4, PREFETCH_WORKERS=2, ASYNC_OUTPUT_WRITES=1.
- Disabled non-E8 flags: GROUPNORM_INPUT_LAYOUT=0, CONV_INPUT_LAYOUT=0,
  CHANNELS_LAST_WEIGHTS=0, KERNEL_FACTORY_INPUT_PACK=0.
- Acceptance: all saved-production equivalence gates must pass; the paired
  slice-total gate must save at least 1.0% before a full 30-session run.
