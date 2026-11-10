#!/usr/bin/env bash
# Train the gated modal dual encoder with the project's Python 3.8 environment.
set -euo pipefail
CQ_TRAIN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CQ_TRAIN_ROOT"
source scripts/activate_train.sh
exec python wheel_legged_gym/scripts/train.py --task nezha2_mine "$@"
