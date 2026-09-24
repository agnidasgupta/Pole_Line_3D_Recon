# E8: bounded ordered asynchronous pipeline

**Update date: 2026-09-24 UTC**
**Status: planned; results are written automatically after validation.**

## Hypothesis

After E7, model execution is already captured in a CUDA Graph. The remaining
end-to-end latency includes CSV parsing, sparse-item preparation, output
durability, and manifest updates. E8 overlaps those CPU-only stages with the
next unchanged inference call.

## Exact contract

E8 must retain E7:

- accepted checkpoint and calibration;
- active-GPU, BF16, batch size 12, channels-last, full model heads;
- 64-cube patch / 48-cube core geometry, fixed padding, active-core order;
- cached reference coordinate channel values, retained safe gather buffers, and
  CUDA Graph replay;
- score fusion, thresholds, labels, Stage-1 NPZ payload, manifests and output
  hierarchy.

A bounded ordered CPU prefetcher may read/parse/build an immutable next
sparse-item while the GPU processes the preceding slice. A single output worker
may write only a deep CPU snapshot, preserves slice order, and completes the
artifact plus manifest before reporting that slice done.

## Automatic decision

The harness runs two E7 controls and two E8 candidates on the representative
157-slice session, requiring all four saved-production equivalence gates to
pass. It then compares mean slice_total_ms, excluding the first slice.
E8 advances to the 30-session gate only when it saves at least 1.0% over the
paired E7 controls. A failure, mismatch, or smaller gain records a rejection
and stops without running the full dataset.

The full gate requires 30 accepted sessions, zero failures, and 30
saved-production equivalence reports. The generated E8_RESULT.md contains
the UTC dates, run roots, exact-equivalence status, timing comparison, and
accept/reject decision. It is the only E8 result file intended for Git.
