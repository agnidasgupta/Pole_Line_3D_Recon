# Archived V6.2 workflows

The old root Python and shell entry points now live here. Use explicit paths, for example `python legacy/v62/infer_v62_stage1_stage2.py --help` or `bash legacy/v62/run_v62_inference.sh`. Shell launchers set their working directory to this folder so sibling commands resolve.

`Dockerfile` and `requirements.txt` preserve the historical V6.2 environment; build with this directory as the Docker context. These workflows still require their historical data, checkpoints and deployment configuration. Their infrastructure operations were not exercised during root cleanup.

For current V4 inference/reconstruction, use the package commands described in the [root README](../../README.md).
