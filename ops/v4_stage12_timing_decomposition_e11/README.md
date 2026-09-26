# E11 — paired Stage 1 → Stage 2 timing decomposition

E11 follows E9 and E10. Both prior experiments preserved exact Stage 1 payloads and exact Stage 2 durable outputs, but missed the required production performance gain. E11 performs two fresh representative-session pairs only; it never starts a full-30 run.

The disk control retains the accepted Stage 1 artifact and Stage 2 load path. The candidate uses E10's direct persistent processor path. Both use the accepted checkpoint, calibration, V10 profile, `active_gpu`, batch size 12, CUDA Graph off, one CPU thread/refiner worker, and unchanged Stage 2 geometry/electrical rules.

Each pair requires exact Stage 1 payloads and an exact Stage 2 output comparison. The emitted report separates Stage 1 artifact/manifest writes, Stage 2 artifact load, reconstruction, primary output writes, audit writes, manifest updates, timing-row writes, and residual callback time. Exactness diagnostics are excluded from measured callback totals.

`FULL30_ELIGIBLE` means only that both representative pairs were exact and showed at least a 2% end-to-end mean gain. It does not launch a full run or alter production. Any other result is `STOP_NO_CREDIBLE_FULL30_GAIN`.
