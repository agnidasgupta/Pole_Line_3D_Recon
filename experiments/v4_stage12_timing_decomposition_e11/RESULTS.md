# E11 Stage 1 → Stage 2 timing decomposition

**Recorded: 2026-09-26 UTC. Decision: NOT ACCEPTED for performance.**

E11 ran two fresh 157-slice representative pairs with the accepted V4
checkpoint, calibration, Stage 2 profile, `active_gpu`, batch size 12, CUDA
Graph off, one CPU thread, and serial refiners. Stage 1 was exact on all 312
timed slices. Both Stage 2 comparisons passed: 1,731 files per repeat and zero
differences.

| Measure | Disk control | Persistent in-memory candidate |
| --- | ---: | ---: |
| End-to-end mean | 1,486.925 ms | 1,557.668 ms |
| Difference | — | **+70.743 ms (+4.758% slower)** |

The disk control's Stage 1 artifact/manifest write and Stage 2 artifact load
totaled about 17.769 ms/slice. The candidate's Stage 2 process time was 71.167
ms/slice higher, so the in-memory transport did not recover its own overhead.

**Conclusion: keep the current disk-backed Stage 1 → Stage 2 handoff.** E11 is
a bounded pre-gate, not a full-30 acceptance run; it closes this transport
optimization path. Protected production `v4` is unchanged.

- [Rendered timing report](results/E11_RESULT_20260926.md)
- [Raw numeric summary](results/E11_RESULT_20260926.json)
