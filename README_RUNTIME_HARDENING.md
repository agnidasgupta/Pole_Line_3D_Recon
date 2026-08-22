# V6.2 teacher-recall runtime hardening

This patch fixes two confirmed inference crashes and hardens later stages against the same class of empty-file/path errors.

## Confirmed inference fixes

1. Every nested inference CSV parent directory is created before `to_csv`.
2. Strict-GT confusion-matrix updates are skipped when a file has a label column but zero valid GT rows after filtering. The inference output for that file is still retained.
3. Empty/header-only source CSVs are skipped with an audit entry instead of crashing the entire run.
4. Empty pole/line/vertex/component outputs are written with headers rather than headerless files.
5. `skipped_missing_sources.csv` and `inference_manifest.csv` are written through a parent-safe writer.

## Stage-2 hardening

Stage-2 component mining now safely resumes past old headerless empty cache CSVs and writes header-bearing empty cache files for slices with no candidates.

## Stage-3 hardening

1. Reconstruction validates that `inference_manifest.csv` exists, is readable, has required columns, and has at least one row before processing.
2. Stage-2 pole/line/vertex readers already catch missing, zero-byte, headerless, and malformed object CSVs; this behavior is retained.
3. Empty Stage-3 audit CSVs now always contain headers.
4. Empty/malformed optional centers CSV input is handled without an uninformative pandas crash.
5. The reconstruction wrapper requires a non-empty inference manifest before launch and backs up any prior reconstruction directory, including an incomplete one, before a fresh Stage-3 rerun.

## Orchestration hardening

`run_v62_teacher_all.sh` now skips stages that already have their completion markers, so it can be used as a stage-aware continuation command after an interrupted run.

`run_v62_teacher_inference.sh` now writes `INFERENCE_FAILED.json` on failure and clears it after successful completion.
