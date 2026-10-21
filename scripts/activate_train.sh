#!/usr/bin/env bash
# Source this file from Bash to activate the project's Python 3.8 training environment.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'Run: source scripts/activate_train.sh\n' >&2
    exit 1
fi
_CQ_TRAIN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "$_CQ_TRAIN_ROOT/.venv/bin/python" ]]; then
    printf 'Missing .venv; see TRAINING_ENV.txt.\n' >&2
    unset _CQ_TRAIN_ROOT
    return 1
fi
source "$_CQ_TRAIN_ROOT/.venv/bin/activate" || return 1
# Avoid mixing ROS Python 3.10 packages with Isaac Gym's Python 3.8 bindings.
unset PYTHONPATH
_CQ_PYTHON_BASE="$("$VIRTUAL_ENV/bin/python" -c 'import sys; print(sys.base_prefix)')" || return 1
export LD_LIBRARY_PATH="$_CQ_PYTHON_BASE/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_EXTENSIONS_DIR="$_CQ_TRAIN_ROOT/.venv/torch_extensions"
export MPLCONFIGDIR="$_CQ_TRAIN_ROOT/.venv/matplotlib"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$MPLCONFIGDIR"
unset _CQ_TRAIN_ROOT _CQ_PYTHON_BASE
