# E8 result

- Update date (UTC): 2026-09-24
- Result created (UTC): 2026-09-24T01:47:03Z
- Contract: E7 plus bounded CPU prefetch/prepare and one ordered async writer only.
- Full run: 30 accepted sessions, zero failures, 30 saved-production equivalence reports.
- Full root: /workspace/voxel_poleline/outputs/poleline_voxel_run_session_groups/v4_production/stage1_opt2_experiments/20260924T010322Z_e8_io_pipeline_full30

## Paired timing gate

~~~~json
{
  "candidate_mean_slice_total_ms": 106.51044385578136,
  "candidates": [
    {
      "excluded_first_slices": 1,
      "means_ms": {
        "csv_read_ms": 46.066509076931304,
        "d2h_gather_ms": 66.43268024999881,
        "slice_total_ms": 106.09902532695077,
        "sparse_item_prep_ms": 14.210361256398475,
        "stage1_artifact_write_ms": 9.256124666654987,
        "stage1_manifest_write_ms": 7.346613403820484,
        "stage1_wall_ms": 86.305740185903
      },
      "run_root": "/outputs/poleline_voxel_run_session_groups/v4_production/stage1_opt2_experiments/20260924T005901Z_e8_io_pipeline_r1",
      "timed_slices": 156,
      "timing_files": 1
    },
    {
      "excluded_first_slices": 1,
      "means_ms": {
        "csv_read_ms": 45.99606921152745,
        "d2h_gather_ms": 66.30571190386584,
        "slice_total_ms": 106.92186238461196,
        "sparse_item_prep_ms": 14.161645839775561,
        "stage1_artifact_write_ms": 9.651423935909593,
        "stage1_manifest_write_ms": 7.487444019244112,
        "stage1_wall_ms": 86.6205442435983
      },
      "run_root": "/outputs/poleline_voxel_run_session_groups/v4_production/stage1_opt2_experiments/20260924T010201Z_e8_io_pipeline_r2",
      "timed_slices": 156,
      "timing_files": 1
    }
  ],
  "control_mean_slice_total_ms": 149.97829225321243,
  "controls": [
    {
      "excluded_first_slices": 1,
      "means_ms": {
        "csv_read_ms": 41.1277429166906,
        "d2h_gather_ms": 71.98346087820995,
        "slice_total_ms": 150.38411538462418,
        "sparse_item_prep_ms": 13.228114448735047,
        "stage1_artifact_write_ms": 7.1980381217990255,
        "stage1_manifest_write_ms": 6.027830384610775,
        "stage1_wall_ms": 81.3140649102549
      },
      "run_root": "/outputs/poleline_voxel_run_session_groups/v4_production/stage1_opt2_experiments/20260924T005721Z_e7_e8_control_r1",
      "timed_slices": 156,
      "timing_files": 1
    },
    {
      "excluded_first_slices": 1,
      "means_ms": {
        "csv_read_ms": 40.91591633331786,
        "d2h_gather_ms": 71.96827799359886,
        "slice_total_ms": 149.57246912180065,
        "sparse_item_prep_ms": 12.96565779487024,
        "stage1_artifact_write_ms": 7.1911004423117655,
        "stage1_manifest_write_ms": 5.861142397427498,
        "stage1_wall_ms": 81.1852896346123
      },
      "run_root": "/outputs/poleline_voxel_run_session_groups/v4_production/stage1_opt2_experiments/20260924T010021Z_e7_e8_control_r2",
      "timed_slices": 156,
      "timing_files": 1
    }
  ],
  "minimum_save_percent": 1.0,
  "saved_percent": 28.982759934379782,
  "status": "PASS"
}
~~~~

## Full timing summary

~~~~json
{
  "excluded_first_slices": 1,
  "means_ms": {
    "csv_read_ms": 75.2240836355369,
    "d2h_gather_ms": 83.14617293390425,
    "slice_total_ms": 141.81761370644278,
    "sparse_item_prep_ms": 22.726338961192354,
    "stage1_artifact_write_ms": 11.220539860852444,
    "stage1_manifest_write_ms": 10.181877875300378,
    "stage1_wall_ms": 115.84628013701784
  },
  "run_root": "/outputs/poleline_voxel_run_session_groups/v4_production/stage1_opt2_experiments/20260924T010322Z_e8_io_pipeline_full30",
  "timed_slices": 3737,
  "timing_files": 30
}
~~~~
