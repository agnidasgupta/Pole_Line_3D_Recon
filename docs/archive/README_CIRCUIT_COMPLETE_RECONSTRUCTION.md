# V6.2 circuit-complete Stage-3 patch

This patch changes reconstruction only. Stage 1/2 outputs are reused.

Key changes:
1. Pole heights are constrained by robust session height bounds (10th-90th percentile plus a small margin, capped around the session median). An attachment that would require an implausibly tall pole is rejected rather than stretching the pole.
2. Fragmented/floating line tracks can be merged into a complete span when two distinct poles bracket a monotonic, nearly collinear, Z-consistent path. The path is acyclic and lane-preserving, so parallel conductors are not connected with rungs.
3. Hidden poles require two non-parallel partial spans, each already anchored at its opposite end to a distinct reconstructed pole. The two unsupported rays must converge inside actually observed slice support and within the valid 10-450 ft span range. A single outgoing line into a discarded/missing slice can never create a pole.
4. After hidden-pole inference, span completion runs again so the same supporting line tracks can be carried through to the inferred pole and become complete spans.

New audit file: `span_completion_paths.csv`.
