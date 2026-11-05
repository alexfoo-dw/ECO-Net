#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT="${1:?Usage: bash scripts/setup_dino.sh <CutLER-commit-hash>}"
mkdir -p third_party
if [[ ! -d third_party/CutLER/.git ]]; then
    git clone https://github.com/facebookresearch/CutLER.git third_party/CutLER
fi
git -C third_party/CutLER checkout --detach "$COMMIT"
git -C third_party/CutLER rev-parse HEAD > third_party/CUTLER_REVISION.txt
