# Result: `v4_stage3_incremental`

**Status: validated experiment.**

The incremental Stage-3 fragment-join experiment completed a **30-session
exact-equivalence regression**. It preserves the Stage-3 contract: only
parametric Stage-2 poles and compact line polylines from the rolling `[S-9,S]`
(450 ft) window are used; Stage 3 does not access voxel data and the output
schema is unchanged.

The corresponding accepted experiment tag is `v4-stage3-incjoin-exp1-eq`.

This result validates equivalence of the incremental approach; it does not by
itself claim a universal throughput improvement for every session topology.

