# E9: direct Stage-1 to Stage-2 handoff

**Updated: 2026-09-25 UTC. Current decision: NOT ACCEPTED.**
The full-30 in-memory candidate remains incomplete; see [E9 results](../../experiments/v4_stage12_inmemory_handoff_e9/RESULTS.md).

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
The Stage-1 control heartbeat checks per-slice progress, session and export logs,
accepted markers and comparison reports as well as the outer driver log.
Candidate sessions retry once with `RESUME=0` and atomic replacement. A failed
control or exact-output comparison stops the run and packages an incomplete
`NOT_ACCEPTED` diagnostic record. No source-code repair or production adoption
is automatic. To restart after fixing a fault, rerun the controller with a new
UTC stamp; previous roots are retained.

## 2026-09-24 interrupted full-30 control

The `20260924T180346Z` harness stopped at session 10/30 because its watchdog
tracked only the outer driver log, which was quiet while session 10 wrote
progress at 18:39:22 UTC. It sent TERM at 18:45:08 UTC, less than six minutes
after that progress update. The two representative output comparisons passed
(1,731 files each, zero differences); nine full-control sessions were accepted.
No full-30 candidate or timing decision was produced. **Decision: NOT ACCEPTED
(incomplete experiment).** This is not evidence of changed inference quality.

The repaired controller watches the actual per-session activity. It supports a
bounded resume of the original Stage-1 control by setting
`E9_FULL30_CONTROL_STAMP=20260924T182337Z` when launching a new E9 harness.
The original control runs with `RESUME=1` and must still pass all 30 accepted
sessions and the production-equivalence marker. The representative pairs run
again under the fresh E9 harness. `--self-test` now also checks the shell
watchdog's progress-file timestamp probe.

## 2026-09-25 Stage-1 reproducibility gate

The saved-control diagnostic compared all 157 Stage-1 payloads in the affected
session. One accepted probe and one E9 control matched production exactly.
The failed r2 and full-30 controls each changed 259,293 pole-score values,
301,220 line-score values, and 76,967 objectness values. Eight predicted
labels changed on six slices: two production positives were lost, one switched
class, and five were newly predicted (unverified, not ground-truth false
positives). The exact-score and prediction contract fails. No full-30 E9
latency gain can be claimed. Details and the raw comparison are in the
[dated result](../../experiments/v4_stage12_inmemory_handoff_e9/RESULTS.md).

Run a fresh representative isolation before E9 again. The isolation compares
CUDA Graph off and on twice with the same checkpoint, calibration, BF16 batch
12, channels-last layout, full heads, thresholds, and all other E7 flags. It
sets `RESUME=0` so each control is fresh; it does not alter V4 production.
Use the committed script on an isolated worktree at Nebius:

```bash
REPO=/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_stage1_electrical_v10
git -C "$REPO" fetch github v4-stage12-inmemory-handoff-e9
E9_WORKTREE="/workspace/voxel_poleline/Pole_Line_3D_Recon_e9_isolation_$(date -u +%Y%m%dT%H%M%SZ)"
git -C "$REPO" worktree add --detach "$E9_WORKTREE" FETCH_HEAD
printf '%s\n' "$E9_WORKTREE" > /home/agni/LATEST_E9_ISOLATION_WORKTREE.txt
LOG=/home/agni/e9_stage1_isolation.log
nohup env REPO="$E9_WORKTREE" bash "$E9_WORKTREE/ops/v4_stage12_inmemory_handoff_e9/run_e9_stage1_isolation.sh" --run > "$LOG" 2>&1 < /dev/null &
```

Monitor without host Python, including after reconnecting:

```bash
E9_WORKTREE=$(cat /home/agni/LATEST_E9_ISOLATION_WORKTREE.txt)
REPO="$E9_WORKTREE" bash "$E9_WORKTREE/ops/v4_stage12_inmemory_handoff_e9/run_e9_stage1_isolation.sh" --monitor
```

After `STATE=ALL_FOUR_EXACT` or `STATE=STAGE1_REPRO_FAILURE`, download only
the small archive named `UPLOAD_ARCHIVE` in the launch log. It contains the
four exact-payload JSON reports, run configuration, source/asset hashes and
logs; it contains no model or NPZ files. Review the four controls before
selecting the same exact inference mode for both E9 arms. Graph-on previously
failed, so two new Graph-on passes do not clear that history. Only if both
fresh Graph-off controls match production should Graph-off be tried in both
arms of a new E9 validation run; all full-30 Stage-1, Stage-2, and timing
checks still apply. No full-30 run starts automatically.
