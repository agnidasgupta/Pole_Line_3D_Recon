# E8 repair: ordered asynchronous I/O pipeline

**Update date: 2026-09-24 UTC**
**Status: planned after the rejected external-core control.**

## Reason for repair

The first E8 attempt imported the complete H100 Opt2 core. Its E7 control
reproduced the known pole-score drift of 0.00019707530736923218, so the control
was not production-equivalent and E8 was not assessed.

This repair starts from the accepted E7 branch and keeps its exact
v4_realtime_core_opt2.py and E7 self-test. It imports only:

- bounded CPU CSV read/parse/sparse-item prefetch;
- one ordered CPU output writer using deep snapshots;
- launcher plumbing and automatic timing/result reporting.

## E8 scope

The repaired candidate enables PREFETCH_INPUTS=1, PREFETCH_DEPTH=4,
PREFETCH_WORKERS=2, and ASYNC_OUTPUT_WRITES=1. It deliberately sets
PREPARE_CORE_SCHEDULE=0, so E7's accepted core scheduling and GPU inference
implementation are untouched.

The two repeated E7 controls and two E8 candidates all run against the accepted
production baseline with score_atol=0. The full 30-session candidate begins only
after all four controls/candidates are exact and E8 saves at least 1.0% mean
slice_total_ms, excluding the first slice.
