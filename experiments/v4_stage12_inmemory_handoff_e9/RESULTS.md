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

**Next gate:** Run the representative Stage-1 isolation in
[`ops/v4_stage12_inmemory_handoff_e9/run_e9_stage1_isolation.sh`](../../ops/v4_stage12_inmemory_handoff_e9/run_e9_stage1_isolation.sh).
It runs fresh Graph-off/Graph-on controls twice with identical other flags,
records source/asset hashes, and compares every saved score and predicted label
against production. It does not run E9 or change production. A failing mode
must not be used for either E9 arm. Because Graph-on controls already failed,
two new Graph-on passes alone cannot clear that history. If both fresh
Graph-off controls pass exactly, use Graph-off for both arms in a *new* E9
validation run; otherwise investigate the source/configuration difference
before running 30 sessions. The isolation never accepts E9: every Stage-1,
Stage-2, coverage, and timing gate in the E9 README still applies.
