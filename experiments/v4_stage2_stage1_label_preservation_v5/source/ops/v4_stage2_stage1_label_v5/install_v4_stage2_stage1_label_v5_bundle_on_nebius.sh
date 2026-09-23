#!/usr/bin/env bash
set -euo pipefail

BUNDLE="${BUNDLE:-/workspace/voxel_poleline/deploy/Pole_Line_3D_Recon_v4_stage2_stage1_label_v5.bundle}"
BRANCH="${BRANCH:-v4-stage2-stage1-label-preservation-v5}"
PRODUCTION_TAG="${PRODUCTION_TAG:-v4.0.1-production-ops}"
EXP_REPO="${EXP_REPO:-/workspace/voxel_poleline/Pole_Line_3D_Recon_v4_stage2_stage1_label_v5}"

fail() { echo "ERROR: $*" >&2; exit 1; }
[ -f "$BUNDLE" ] || fail "bundle missing: $BUNDLE"
[ -f "${BUNDLE}.commit" ] || fail "bundle commit record missing: ${BUNDLE}.commit"
if [ -f "${BUNDLE}.sha256" ]; then
  EXPECTED=$(awk '{print $1}' "${BUNDLE}.sha256")
  ACTUAL=$(sha256sum "$BUNDLE" | awk '{print $1}')
  [ "$EXPECTED" = "$ACTUAL" ] || fail "bundle SHA256 mismatch"
  echo "BUNDLE SHA256 OK"
fi
EXPECTED_COMMIT=$(tr -d '[:space:]' < "${BUNDLE}.commit")

git bundle list-heads "$BUNDLE" | grep -F "refs/heads/$BRANCH" >/dev/null \
  || fail "branch not present in bundle: $BRANCH"

if [ -e "$EXP_REPO" ] && [ ! -d "$EXP_REPO/.git" ] && [ ! -f "$EXP_REPO/.git" ]; then
  OLD="${EXP_REPO}.partial_$(date -u +%Y%m%dT%H%M%SZ)"
  mv "$EXP_REPO" "$OLD"
  echo "Moved non-Git partial path to $OLD"
fi

if [ ! -e "$EXP_REPO" ]; then
  mkdir -p "$(dirname "$EXP_REPO")"
  git clone --branch "$BRANCH" --single-branch "$BUNDLE" "$EXP_REPO"
else
  DIRTY=$(git -C "$EXP_REPO" status --porcelain)
  [ -z "$DIRTY" ] || fail "existing experiment repository has uncommitted changes"
  git -C "$EXP_REPO" fetch --force "$BUNDLE" \
    "refs/heads/$BRANCH:refs/remotes/bundle/$BRANCH"
  CURRENT=$(git -C "$EXP_REPO" rev-parse HEAD)
  git -C "$EXP_REPO" branch "backup/pre_stage1_label_v5_$(date -u +%Y%m%dT%H%M%SZ)" "$CURRENT"
  git -C "$EXP_REPO" switch "$BRANCH"
  git -C "$EXP_REPO" reset --hard "refs/remotes/bundle/$BRANCH"
fi

if ! git -C "$EXP_REPO" rev-parse "${PRODUCTION_TAG}^{commit}" >/dev/null 2>&1; then
  git -C "$EXP_REPO" fetch "$BUNDLE" \
    "refs/tags/$PRODUCTION_TAG:refs/tags/$PRODUCTION_TAG"
fi

ACTUAL_COMMIT=$(git -C "$EXP_REPO" rev-parse HEAD)
echo "Expected: $EXPECTED_COMMIT"
echo "Actual:   $ACTUAL_COMMIT"
[ "$EXPECTED_COMMIT" = "$ACTUAL_COMMIT" ] || fail "Nebius source commit mismatch"
[ -z "$(git -C "$EXP_REPO" status --porcelain)" ] || fail "repository is dirty after installation"

echo "V4_STAGE2_STAGE1_LABEL_BUNDLE_INSTALL_OK"
echo "repository=$EXP_REPO"
echo "branch=$(git -C "$EXP_REPO" branch --show-current)"
echo "commit=$ACTUAL_COMMIT"
