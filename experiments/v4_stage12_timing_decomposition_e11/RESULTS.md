# E11 Stage 1 → Stage 2 timing decomposition

**Recorded: 2026-09-26 UTC. Decision: NOT ACCEPTED for performance.**

E11 is the final bounded pre-gate for the Stage 1 → Stage 2 in-memory
handoff. It ran two fresh 157-slice representative pairs using the accepted
V4 checkpoint, calibration, Stage 2 profile, `active_gpu`, batch size 12,
CUDA Graph off, one CPU thread, and serial refiner workers. The disk control
writes the Stage 1 artifact and reloads it for Stage 2; the candidate passes
the exact payload to a persistent in-memory Stage 2 processor. Production
`v4` is unchanged.

## Quality gate

| Check | Observed result |
| --- | --- |
| Timed slices | 312 (first slice of each representative run excluded) |
| Stage 1 payloads | 312/312 exact |
| Stage 2 output comparisons | 2/2 PASS |
| Files compared | 1,731 per repeat; zero differences |
| Reconstruction schemas/rules | unchanged |

The exactness diagnostics were outside the measured callback intervals.

## Performance result

| Measure | Disk control | Persistent in-memory candidate |
| --- | ---: | ---: |
| Stage 2 callback mean | 1,333.644 ms | 1,409.506 ms |
| End-to-end mean | 1,486.925 ms | 1,557.668 ms |
| End-to-end difference | — | **+70.743 ms (+4.758% slower)** |

This confirms that the current disk-backed Stage 1 → Stage 2 handoff is
faster for the accepted V4 production configuration. The removable disk
artifact/manifest write plus Stage 2 artifact load measured about **17.769
ms/slice** in the control. In the candidate, Stage 2 `process` time was
**71.167 ms/slice higher** and audit-output writing was **6.995 ms/slice
higher**, outweighing the transport saving.

E9, E10, and E11 all preserved their tested outputs but did not produce a
performance benefit. E11 therefore closes this transport-optimization path:
**do not start a full 30-session handoff run and do not modify protected
production `v4`.**

- [Rendered E11 timing report](results/E11_RESULT_20260926.md)
- [Raw E11 numeric summary](results/E11_RESULT_20260926.json)
