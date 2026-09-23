# Promotion rule

Opt1 is an experiment, not a replacement for the accepted Stage 1.

Promotion is allowed only after one complete 30-session Nebius run has:

1. `STAGE1_OPT_EQUIVALENT_COMPLETE.txt`;
2. exactly 30 `status/*.stage1.ok` files;
3. exactly 30 `status/*.quality_equivalence.json` reports;
4. zero predicted-label mismatches and unchanged Stage 1 metrics in every report;
5. all score differences within `1e-4`;
6. no runtime failure, timeout, CUDA out-of-memory event, or possible-stall condition;
7. a material improvement in the all-session `baseline_comparable_total_ms` shown in
   `STAGE1_TIMING_SESSION_AVERAGES.txt`.

If speed does not improve, retain the accepted implementation. Never loosen the quality
gate to promote a faster result.
