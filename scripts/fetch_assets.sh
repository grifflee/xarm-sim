#!/usr/bin/env bash
# Fetch large binary assets that are not tracked in git.
#
# assets/lab_aligned.ply is the baked-alignment gaussian splat of the lab scene
# (~313 MiB). It is gitignored (see .gitignore), so a fresh clone has to pull it
# from the GitHub release before any sim env can be constructed -- a missing file
# is a hard FileNotFoundError in TaskEnv, not a silent fallback.
#
# Usage: scripts/fetch_assets.sh
set -euo pipefail

REPO="mhyatt000/xarm-sim"
TAG="assets-v1"
NAME="lab_aligned.ply"
EXPECTED_MD5="13a21b6e3df686d2cc7169f52f0879a3"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/assets/$NAME"

md5_of() { md5sum "$1" | cut -d' ' -f1; }

if [ -f "$DEST" ] && [ "$(md5_of "$DEST")" = "$EXPECTED_MD5" ]; then
    echo "ok: $DEST already matches $EXPECTED_MD5"
    exit 0
fi

if ! command -v gh >/dev/null 2>&1; then
    echo "error: gh CLI is required to download $NAME from $REPO release $TAG" >&2
    exit 1
fi

echo "downloading $NAME from $REPO release $TAG ..."
mkdir -p "$ROOT/assets"
gh release download "$TAG" --repo "$REPO" --pattern "$NAME" --dir "$ROOT/assets" --clobber

ACTUAL_MD5="$(md5_of "$DEST")"
if [ "$ACTUAL_MD5" != "$EXPECTED_MD5" ]; then
    echo "error: md5 mismatch for $DEST (got $ACTUAL_MD5, want $EXPECTED_MD5)" >&2
    exit 1
fi
echo "ok: $DEST verified ($EXPECTED_MD5)"
