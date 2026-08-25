# V4 realtime production acceptance checklist

Do not merge the candidate into the production branch until every required item below is reviewed.

## Stage 1

- Accepted V4 checkpoint/calibration unchanged.
- H100 `full_cpu` reference completes successfully.
- Four-way runtime gate completed on real slices.
- Selected runtime has zero Stage-1 label mismatches versus `full_cpu` and score deltas within tolerance.
- If selected runtime is `active_cpu`, the active-core speed gain is material enough to keep.
- Batch-size sweep completed on the selected runtime.
- Production batch size chosen from measured latency, not memory capacity alone.

## Stage 2

- Stage-2 mining uses exactly the selected Stage-1 runtime mode and batch/precision configuration.
- Stage-2 bundle contains no center/world/GT-overlap model features.
- Stage-2 refiner metrics and selected thresholds reviewed.
- Stage-2 per-slice timing reviewed at P50/P95.

## Stage 3

- Stage 3 runs after every arriving slice.
- For newest sequence S, every snapshot uses only acquired slices in `[S-9, S]`.
- Missing sequence numbers are accepted.
- Future-slice evidence is absent.
- Full-session/offline Stage-3 finalize is disabled by the production runner.
- Fragment joins, span completion, hidden-pole inference, attachments, and output-write timing reviewed.

## End-to-end realtime

- 5-slice clean benchmark passed.
- Full recorded-session benchmark passed.
- `REALTIME_REPLAY_VERIFICATION.json` reports success.
- End-to-end P50/P95 update latency reviewed.
- Stage-3 latency versus rolling-window size reviewed.
- Representative reconstruction CSVs/audits visually/structurally reviewed.
- Compact production-review archive saved locally and shared for final analysis.
