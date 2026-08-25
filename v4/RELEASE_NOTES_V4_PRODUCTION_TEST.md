# V4 production-test release notes

This test release preserves the accepted V4 model behavior while targeting realtime inference/reconstruction latency and recoverability.

Key changes:

- four-way Stage 1 runtime gate with downstream Stage 2 equivalence;
- sparse one-transfer GPU-volume Stage 1 path and fixed-shape active batching;
- expanded GPU/CPU/data-transfer timing;
- atomic durable Stage 1 and Stage 2 boundaries;
- independently executable Stage 1, Stage 2, and Stage 3 runners;
- moved-artifact fallback so downstream stages remain runnable from saved results;
- rolling Stage 3 in-memory cache for realtime handoff while retaining disk authority;
- cached fragment geometry / shared endpoint candidate search;
- redundant post-hidden span completion skipped when no hidden pole is added;
- explicit `9 sequence gaps x 50 ft = 450 ft`, with up to 10 observed centers;
- shell-only Nebius path/session discovery and exact deployment fingerprints;
- detached build, preflight, H100 acceptance, stage runners, and production replay;
- persistent run context, session inventory, logs, completion/failure markers, and code backups;
- review package can be created after either success or failure;
- Mac download helper verifies SHA256 without Python.

The H100 gate and full-session replay are intentionally not pre-declared successful in this release. They must be run on Nebius against the actual checkpoint/calibration/Stage-2 bundle before GitHub V4 is updated.
