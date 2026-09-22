# Containers

The old root Dockerfile has moved to `legacy/v62/Dockerfile`, with build context `legacy/v62/`. For current V4, use the image below; bare `docker build .` is not the V4 build command.

The compatibility image remains `v4/Dockerfile.v4_realtime` with build context `v4/`.
The image installs dependencies; launchers bind-mount source. Updated launchers
mount the whole checkout at `/workspace/poleline_repo` in addition to historical
V4 paths so compatibility entry points can locate the package. For manual runs,
mount the whole checkout and use its original entry points or install it editable.
