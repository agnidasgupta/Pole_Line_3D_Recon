#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/agnidasgupta/Pole_Line_3D_Recon.git}"
BRANCH="${BRANCH:-v4-stage2-gt-autoselect-v3}"
DEST="${DEST:-/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_gt_autoselect_v3}"

[ ! -e "$DEST" ] || { echo "ERROR: destination already exists: $DEST" >&2; exit 1; }
ANON_HOME=$(mktemp -d /tmp/github-anon.XXXXXX)
trap 'rm -rf "$ANON_HOME"' EXIT
mkdir -p "$ANON_HOME/.config"

run_anon_git() {
  env \
    -u GH_TOKEN \
    -u GITHUB_TOKEN \
    -u GIT_ASKPASS \
    -u SSH_ASKPASS \
    HOME="$ANON_HOME" \
    XDG_CONFIG_HOME="$ANON_HOME/.config" \
    GIT_CONFIG_NOSYSTEM=1 \
    GIT_TERMINAL_PROMPT=0 \
    git \
      -c credential.helper= \
      -c core.askPass= \
      -c http.extraHeader= \
      "$@"
}

run_anon_git ls-remote --exit-code --heads "$REPO_URL" "refs/heads/$BRANCH" >/dev/null
run_anon_git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$DEST"

run_anon_git -C "$DEST" fetch origin --tags

echo "PUBLIC_GITHUB_CLONE_OK"
echo "destination=$DEST"
echo "branch=$(git -C "$DEST" branch --show-current)"
echo "commit=$(git -C "$DEST" rev-parse HEAD)"
