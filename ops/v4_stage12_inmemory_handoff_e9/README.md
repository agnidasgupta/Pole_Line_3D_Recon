# E9: direct Stage-1 to Stage-2 handoff

**Update date: 2026-09-24 UTC**
**Status: planned; no performance result is recorded until the automated full-30 gate completes.**

E9 changes only the handoff transport. The accepted E7 Stage-1 inference core,
checkpoint, calibration, BF16 batch-12 configuration, scores, labels and
Stage-2 V10 reconstruction rules are unchanged.

For the candidate, the Stage-1 payload is verified byte-exact in memory against
the accepted production Stage-1 artifact, then supplied directly to V10 Stage 2.
Candidate Stage-1 NPZ/JSON artifacts are not written. Stage-2 continues to
write its existing atomic CSV/JSON outputs and manifests. The control performs
the same inference, writes the standard Stage-1 artifacts, then has Stage 2
reload them.

The controller verifies SHA-256 for the accepted production checkpoint,
calibration, V10 bundle, and selected electrical profile before running.
Only these pinned assets are used. Control and candidate both pin the existing
refiner to one worker to remove the known ULP-level CSV nondeterminism.

Acceptance requires two repeated representative controls/candidates and the
full 30-session run to have zero Stage-1 array differences, 32/32 Stage-2
durable-output comparisons, 30/30 sessions, 3,738/3,738 slices and identical
complete slice keys.
It compares semantic CSV files byte for byte; Stage-2 manifests and JSON
summaries are compared after removing only path/handoff-location fields.
The full result is accepted only if these gates pass and combined mean
per-slice time improves by at least 1%. It reports raw slice timing data,
mean/P50/P95 and a breakdown for CSV read, sparse prep, inference, Stage-1
disk writes (control), Stage-2 load/reconstruction, and Stage-2 output work.
The first slice of each session is excluded from the steady-state comparison.

On a runtime error or stale progress heartbeat (default 600 seconds, checked
every 20–30 seconds), the controller records GPU, container and log diagnostics.
Candidate sessions retry once with `RESUME=0` and atomic replacement. A failed
control or exact-output comparison stops the run and packages an incomplete
`NOT_ACCEPTED` diagnostic record. No source-code repair or production adoption
is automatic. To restart after fixing a fault, rerun the controller with a new
UTC stamp; previous roots are retained.
