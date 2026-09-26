# E10: persistent Stage 2 processor

**Experiment prepared 2026-09-25 UTC. Decision: pending full validation.**

E9's full 30-session handoff preserved the exact Stage 1 arrays on 3,738/3,738
slices and all checked Stage 2 outputs, but its measured combined mean rose
from 2,244.981 to 2,303.783 ms (+58.802 ms, 2.619% slower). E9 invoked the
whole V10 Stage 2 command for each slice and repeatedly rebuilt a one-row
Stage 1 manifest. E10 holds one Stage 2 processor for the session and calls
its original `process(item, pred, id, seq)` directly. The writer reproduces the
V10 disk runner's atomic output, voxel audits, manifest, and completion
summary. It still never writes candidate Stage 1 NPZ/JSON to disk.

The fresh disk control and candidate both use the accepted V4 checkpoint,
calibration, BF16 active GPU inference, batch 12, unchanged thresholds and
scores, the selected V10 bundle/profile and serial refiner. CUDA Graph is
disabled in both arms because earlier E9 controls showed occasional numerical
differences when it was enabled. This is a matched test of transport and
orchestration; it is not a changed prediction rule or a new model.

The controller runs two repeated representative sessions, then all 30
sessions. Each candidate slice is compared with accepted production Stage 1
NPZ arrays by exact dtype, shape, and bytes; each session's Stage 2 CSVs are
byte compared with its matched control. Stage 2 JSON and manifests are
compared after removing location-only fields. Any difference stops the run.
The required full-run gate is 3,738 slices, 30 sessions, 3,738 exact Stage 1
payloads, and 32/32 Stage 2 comparisons. For performance acceptance, the
mean combined timing must improve at least 2%, at least 16 of the 29 timed
sessions must improve, and P95 must rise no more than 5%. First slice of each
session is excluded only from timing, never from quality checks.

The combined control timing is Stage 1 CSV read, sparse preparation,
inference, Stage 1 artifact and manifest writes, and the Stage 2 disk slice
wall. The candidate timing substitutes its direct Stage 2 callback and output
write for the disk path and includes the measured temporary Stage 1 manifest
write. Production NPZ reload used by the candidate's exact
validation is timed separately and excluded from the candidate performance
figure. These boundaries are analogous but not identical, so any marginal
gain needs a further controlled repeat before production adoption.

The output hierarchy is under
`outputs/v4_stage12_persistent_stage2_e10/<UTC stamp>/` with `control/`,
`candidate/`, `comparisons/`, `diagnostics/`, `status/`, and `summaries/`.
The candidate's normal Stage 2 files remain under `stage2/<safe session id>/`.
The result package contains numeric and raw timing summaries, comparison
reports and diagnostics, without Stage 1 NPZ, model, or large reconstruction
CSVs. The final decision is in `summaries/E10_RESULT.md` and `.json`.

This branch remains experimental. `v4` production and its accepted outputs
are not modified by the test.
