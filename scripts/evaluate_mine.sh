#!/usr/bin/env bash
# Load one trained policy and evaluate every selected terrain in parallel.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source "$project_root/scripts/activate_train.sh"
exec "$VIRTUAL_ENV/bin/python" -u wheel_legged_gym/scripts/evaluate.py "$@"
