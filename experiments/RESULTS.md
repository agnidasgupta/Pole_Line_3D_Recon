# V4 experiment results index

This index makes the evidence recorded for every retained V4 experiment readable
without materializing its source branch or running an experiment script. A
`RESULTS.md` within each experiment directory contains its source reference and
the concise outcome known at the pinned revision.

The registry remains the source-of-truth for provenance only:
[`REGISTRY.tsv`](REGISTRY.tsv).

| Experiment | Recorded result |
|---|---|
| [harpreet_v4_h100_opt](harpreet_v4_h100_opt/RESULTS.md) | **Validated**: Stage 1 11.3% lower steady latency; Stage 2 exact after serial refiner evaluation. |
| [v4_stage1_inference_opt1](v4_stage1_inference_opt1/RESULTS.md) | Profiling/inference experiment archived; no final acceptance result recorded here. |
| [v4_stage1_inference_opt2](v4_stage1_inference_opt2/RESULTS.md) | **Validated experiment**: production-equivalence checks passed through the 30-session, 3,738-slice E7 run. |
| [v4_stage2_stage1_electrical_v10](v4_stage2_stage1_electrical_v10/RESULTS.md) | Stage-2 reconstruction contract source; no separate result record was preserved. |
| [v4_stage2_stage1_electrical_v10_opt1](v4_stage2_stage1_electrical_v10_opt1/RESULTS.md) | Optimized reconstruction experiment; no independently accepted benchmark is recorded here. |
| [v4_stage3_incremental](v4_stage3_incremental/RESULTS.md) | **Validated experiment**: 30-session exact-equivalence regression recorded. |
| [v4_stage3_incremental_chain](v4_stage3_incremental_chain/RESULTS.md) | Source snapshot retained; no separate result record was preserved. |
| [all other retained V4 experiments](#archived-experiments-without-a-recorded-result) | Source and author are preserved, but no validated numeric or equivalence result was supplied for publication. |

## Reading the result status

- **Validated** means the result below was observed in the recorded run and
  passed its stated equivalence gate.
- **Archived** means the experiment source is intentionally preserved, but this
  clean repository does not claim an outcome that has not been validated and
  recorded.
- Stage-1 label metrics are interpreted under the established contract:
  positive pole/line labels are trusted; missing labels do not establish that a
  pole or line prediction is false.

## Archived experiments without a recorded result

The following pinned source experiments remain runnable through their individual
README files and `REGISTRY.tsv`, but their key outcome was not present in the
accepted validation records used to build this index:

- `v4_recon_quality_enhancements`
- `v4_stage12_unity_native_v10`
- `v4_stage2_bundle_consensus_v7`
- `v4_stage2_bundle_consensus_v7_diagnostics`
- `v4_stage2_gt_autoselect_v3`
- `v4_stage2_native_line_recall_v2`
- `v4_stage2_recall_stage3_quality_v1`
- `v4_stage2_residual_union_v6`
- `v4_stage2_stage1_exact_v8`
- `v4_stage2_stage1_label_preservation_v5`
- `v4_stage2_stage1_track_join_v9`

Their `RESULTS.md` documents state this explicitly, which is preferable to
presenting an unsupported result as accepted.
