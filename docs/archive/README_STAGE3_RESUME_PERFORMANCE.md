# V6.2 Stage-3 resumable/performance hardening

This patch is intended for a Stage-3 reconstruction interrupted or apparently hung on dense sessions.

## Why the previous Stage 3 could appear stuck

The previous fragment joiner compared every line fragment with every other fragment, twice (strict and detached-bridge passes). It also scanned every fragment again for each candidate when checking whether a join skipped an intervening fragment. Dense sessions with thousands to tens of thousands of Stage-2 line segments could therefore become extremely expensive.

## Changes

1. Fragment candidate generation now uses cKDTree endpoint spatial indexing and the same slice-gap/distance limits as the original geometric test.
2. The intervening-fragment guard uses a midpoint cKDTree rather than scanning the entire session for every candidate.
3. Pole-pair span completion uses a pole cKDTree and track representative spatial index to avoid testing distant tracks against every pole pair.
4. Hidden-pole endpoint pairing uses a cKDTree bounded by the maximum possible two-ray extrapolation distance.
5. Stage 3 supports `--resume_sessions`. Sessions with complete `summary.json`, `world_poles.csv`, `conductor_chains.csv`, `conductor_vertices.csv`, and `spans.csv` are loaded and skipped.
6. An interrupted/incomplete session directory is deleted and recomputed; completed session directories are preserved.
7. New per-session audit CSVs are written so future resumes can rebuild detailed global audits without recomputation.
8. CSV outputs are written atomically using `.tmp` then rename.
9. Per-session phase timing is logged for fragment joining, span completion, and hidden-pole inference.
10. `resume_session_audit.csv` records whether a resumed legacy session had detailed per-session audits available. Sessions completed before this patch generally do not; their summary/core reconstruction outputs are still loaded and aggregate counts remain correct.

## Resume

Set `RESUME_STAGE3=1` when calling `run_v62_teacher_reconstruction_on_nebius.sh`. The wrapper does not back up/replace the current Stage-3 tree in this mode. It removes only the stale top-level COMPLETED.json and the Python program cleans only an incomplete current session.
