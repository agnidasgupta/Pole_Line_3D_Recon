# E9 Stage 1 → Stage 2 in-memory handoff: current result

**Updated: 2026-09-25 UTC. Decision: NOT ACCEPTED.** The E9 full-30 candidate
has not run to completion. There is no measured E9 speedup and no Stage-2
equivalence result for a completed 30-session candidate.

E9 changes handoff transport: the candidate passes the Stage-1 payload to
Stage 2 in memory and writes the usual Stage-2 outputs. The control writes
Stage-1 artifacts before Stage 2 reloads them. It must preserve the accepted
V4 scores, predictions, reconstruction rules, and complete output coverage.

## Observed Stage-1 control comparisons

These are **one session of 157 slices each**, compared against the accepted
2026-08-25 V4 production Stage-1 payload. Score counts are array values and
must not be interpreted as unique voxels. The exact-score gate is `atol=0`.

| Run | Complete NPZ comparison | Pole score values changed | Line score values changed | Objectness values changed | Max absolute pole / line / objectness difference | Predicted labels changed | Decision |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| `20260924T202241Z_e9_s1_fidelity_probe` | 157/157 | 0 | 0 | 0 | 0 / 0 / 0 | 0 | PASS |
| `20260924T225930Z_e9_control_r1` | 157/157 | 0 | 0 | 0 | 0 / 0 / 0 | 0 | PASS |
| `20260924T230931Z_e9_control_r2` | 157/157 | 259,293 | 301,220 | 76,967 | 0.005798459 / 0.006320387 / 0.003680587 | 8 | FAIL |
| `20260924T182337Z_e9_control_full30`, affected session | 157/157 | 259,293 | 301,220 | 76,967 | 0.005798459 / 0.006320387 / 0.003680587 | 8 | FAIL |

Both failed runs differ in score arrays on **all 157 slices**, beginning with
slice 150, and have the same counts and maxima in this diagnostic. Their
`semantic` arrays differ on **six slices**. Each has eight changed predicted
labels on six slices: **two accepted production positive predictions become
unlabeled, one positive switches pole↔line, and five previously unlabeled
voxels receive a pole/line prediction**. The five additions are *unverified*,
not false positives: missing pole/line ground-truth labels are incomplete.
The two losses and one class change violate the production prediction contract.

The first comparator stopped at a pole-score difference of
`0.00019707530736923218` at slice 150. The later diagnostic enumerated the
complete payload rather than stopping at that first difference. Identical
aggregate fingerprints suggest a repeatable alternate execution path, but
they do **not** prove the two failed runs are byte-identical or establish a
CUDA Graph root cause.

[Raw comparison JSON](results/e9_stage1_repro_20260925T014419Z.json) contains
inventory, all aggregate counts, up to 30 per-slice examples, and errors.
It contains no model weights or NPZ payloads.

## Timing and reconstruction status

The two earlier representative Stage-2 output comparisons reported **1,731
files each and zero differences**. They do not substitute for a completed
full-30 comparison. The watchdog interruption on the first full-30 attempt
and the later exact-score control failures blocked E9 before a valid timing
summary. A synthetic self-test percentage is not a measured E9 gain.

## Fresh Stage-1 isolation (2026-09-25 UTC)

The A-B-A-B isolation used four **fresh** 157-slice runs at source commit
`e6f6ff07ba072401acf420cf4fbd3d4691697bc9`. The accepted checkpoint and
calibration hashes matched the production assets above. All runs used BF16,
batch 12, channels-last, full model heads, `RESUME=0`, and `score_atol=0`;
the only tested setting changed between runs was CUDA Graph replay.

| Run | CUDA Graph | Production score payloads exact | Pole / line / objectness score values changed | Predicted labels changed | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `e9_isolate_graph0_r1` | Off | 157/157 | 0 / 0 / 0 | 0 | PASS |
| `e9_isolate_graph1_r1` | On | 157/157 | 0 / 0 / 0 | 0 | PASS |
| `e9_isolate_graph0_r2` | Off | 157/157 | 0 / 0 / 0 | 0 | PASS |
| `e9_isolate_graph1_r2` | On | 157/157 | 0 / 0 / 0 | 0 | PASS |

In total, **628/628 slice payload comparisons passed**, with zero changed
score values and zero changed labels. The four independent reports show no
missing/extra files or comparison errors. This supports exact reproduction in
these fresh representative runs. It does **not** explain the earlier Graph-on
failures or prove that the `RESUME` flag caused them; the earlier failures
remain evidence of an intermittent alternate execution path. The isolation
did not run Stage 2 or the complete 30-session candidate.

[Machine-readable isolation results](results/E9_STAGE1_ISOLATION_20260925.json)
include all four per-run score and label counts without model or NPZ files.

The next experimental controller uses **CUDA Graph off and `RESUME=0` for both
Stage-1 arms**, so the disk control and in-memory candidate are compared under
the same fresh-run configuration. It refuses `E9_FULL30_CONTROL_STAMP` to
prevent reuse of a previous, potentially non-equivalent control. These are
execution settings only; the accepted checkpoint, calibration, scores,
thresholds, and reconstruction rules are unchanged. The 30-session run must
still pass its per-session exact Stage-1 payload, Stage-2 output, coverage,
and timing gates before E9 can be accepted.

**Current decision: NOT ACCEPTED.** No valid full-30 E9 speedup is available.
