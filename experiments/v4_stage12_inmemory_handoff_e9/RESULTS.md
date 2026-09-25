# E9 Stage 1 → Stage 2 in-memory handoff: current result

**Updated: 2026-09-25 UTC. Decision: NOT ACCEPTED for performance.** The
30-session E9 run completed: Stage 1 was exact on 3,738/3,738 slices and all
32 Stage-2 output comparisons passed. Its measured combined mean was 2.619%
slower than the disk control, below the required 1% saving.

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

## Earlier attempts

The first full-30 attempt stopped under a watchdog, and two earlier Stage-1
controls failed exact prediction checks. The self-test's illustrative
`+4.348%` is not a measured result. A later fresh run completed and is
reported below; the earlier failed controls remain relevant to reproducibility.

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

## Completed 30-session E9 result (2026-09-25 UTC)

| Measure | Disk control | In-memory candidate | Result |
| --- | ---: | ---: | --- |
| Sessions / slices | 30 / 3,738 | 30 / 3,738 | Complete coverage |
| Stage-1 payload checks | — | 3,738/3,738 exact | PASS |
| Stage-2 output comparisons | — | 32/32 PASS | 44,700 compared file checks, zero differences; 41,238 in the full-30 set |
| Timed slices | 3,708 | 3,708 | First slice of each session excluded |
| Combined mean per slice | 2,244.981 ms | 2,303.783 ms | Candidate +58.802 ms; **2.619% slower** |
| Combined P50 | 979.609 ms | 992.216 ms | Candidate +12.606 ms |
| Combined P95 | 8,643.970 ms | 8,847.415 ms | Candidate +203.444 ms |

Stage-2 CSVs were compared byte-for-byte; manifest and JSON comparisons
normalize only location-specific handoff paths. No reconstruction differences
were found.

The disk control's mean Stage-1 artifact write was **10.115 ms**, manifest
write **9.247 ms**, and Stage-2 artifact load **4.584 ms**. Their sum is
**23.947 ms**, about **1.07%** of the control's 2,244.981 ms combined mean.
Stage-2 reconstruction dominated: the control measured **1,977.044 ms** and
the candidate **2,025.778 ms** (+48.734 ms). The respective Stage-2 enclosing
wall times were **2,028.061 ms** and **2,109.040 ms** (+80.979 ms); the
non-reconstruction portion of that envelope increased by about **32.245 ms**.
The heavy Stage-2 work and wrapper overhead outweighed the small removable
file I/O. Paired slice comparisons were slower on 68.9% of the 3,708 timed
slices; 24 of the 29 sessions with timed slices were slower. One single-slice
session has no timed slice after first-slice exclusion.

The Stage-2 algorithm and output files were unchanged by the handoff, so the
+48.734 ms process-time difference is an observed runtime difference, not
evidence of a new reconstruction rule. It may include run-order and CPU-load
variation. The wrapper also calls the Stage-2 driver for each slice and uses a
temporary manifest. The reported candidate combined metric omits its separate
per-slice production-NPZ exactness check and its later Stage-1 manifest write;
therefore it is **not a complete physical end-to-end wall clock** and cannot
be treated as a clean measurement of file I/O alone. Even with this favorable
accounting, it did not improve the combined time.

[Raw numeric summary](results/E9_RESULT_20260925.json) and
[rendered timing report](results/E9_RESULT_20260925.md) retain the control
and candidate means, P50/P95 values, all measured component breakdowns,
coverage, and the acceptance decision. No model or Stage-1 NPZ files are
published here.

**Decision: NOT ACCEPTED for speed.** Output equivalence passed. A future
transport-only experiment should keep one Stage-2 processor alive per session,
feed the exact Stage-1 arrays directly to its existing computation and writer,
and time the same boundaries in both arms. Run exact-output checks outside
the timed region and retain the full 30-session gate.
