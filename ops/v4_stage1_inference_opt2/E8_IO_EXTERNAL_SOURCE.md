# E8 I/O repair source boundary

**Update date: 2026-09-24 UTC**

The repaired E8 branch starts from `v4-stage1-inference-opt2`, the accepted E7
source. It retains that branch's `v4_realtime_core_opt2.py` and
`self_test_v4_stage1_opt2.py` unchanged.

Only the following I/O-oriented files are imported from
`harpreet/v4-h100-opt`:

- `ops/v4_stage1_inference_opt2/run_v4_stage1_opt2.py`
- `ops/v4_stage1_inference_opt2/launch_v4_stage1_opt2_experiment.sh`
- `ops/v4_stage1_inference_opt2/run_v4_stage1_opt2_experiment.sh`
- `v4/v4_input_prefetch.py`
- `v4/v4_output_writer.py`

The repaired runner removes the external core-schedule and kernel-factory
arguments before it calls the exact E7 core. The experiment also disables
`PREPARE_CORE_SCHEDULE`, model-layout flags, and kernel-factory packing.
