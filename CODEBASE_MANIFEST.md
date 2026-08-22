# Codebase manifest

This repository is assembled from the latest V6.2 code in this order:

1. V6.2 teacher-recall base pipeline.
2. Runtime-hardening patch for nested output directories, empty GT metric arrays, empty/header-only component files, and Stage-3 preflight/audits.
3. Resumable/performance Stage-3 patch using spatial indexing and per-session resume.
4. One-session inference/reconstruction timing utilities.

The later patches overwrite the corresponding older files. Therefore the files in this repository represent the consolidated current versions rather than a collection of mutually incompatible patch directories.
