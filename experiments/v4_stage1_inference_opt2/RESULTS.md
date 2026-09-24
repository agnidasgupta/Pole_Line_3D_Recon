# Result: `v4_stage1_inference_opt2`

**Status: validated experiment; not a production merge.**

The recorded Opt2 experiment preserved the accepted Stage-1 inference contract:
checkpoint and calibration, active-GPU BF16, batch size 12, channels-last
layout, full model heads, 64³ patch / 48³ core, score fusion, thresholds, and
output decisions.

The final E7 full-dataset run completed with **30 accepted sessions**, **0
failures**, **30 production-equivalence passes**, and **3,738 timing rows**.
Its CUDA graph implementation captured once and replayed 794 times on the
representative session. Subsequent timing comparison showed the CUDA-graph path
did not provide a stable end-to-end improvement because host result gathering
dominated the remaining latency; it is retained as experimental evidence, not a
production recommendation.

The result is interpreted using the project label rule: known positive
pole/line labels must be preserved, while unlabeled voxels are not evidence of
a false positive.


## E8 I/O repair result (updated 2026-09-24 UTC)

The verified record is [results/E8_IO_RESULT.md](results/E8_IO_RESULT.md).
