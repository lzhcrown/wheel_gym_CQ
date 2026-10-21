#!/usr/bin/env bash
# Run the six chapter 4.5.1 experiments sequentially with identical CLI settings.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash scripts/train_chapter451.sh [options]

  --max_iterations N  Training iterations per experiment (default: 10000)
  --num_envs N        Parallel simulation environments within each experiment (default: 4096)
  --seed N            Shared nonnegative random seed (default: 10)
  --save_interval N   Checkpoint interval in completed iterations (default: 100)
  --headless         Run without a viewer window (already enabled by default)
  --dry-run          Print the six commands without activating Python or training
  --help, -h         Show this help

Each experiment must finish successfully before the next starts.
A failed experiment or Ctrl+C stops the batch; existing results are preserved.
HELP
}

fail() {
    printf 'Error: %s\n' "$1" >&2
    exit 2
}

iterations=10000
num_envs=4096
seed=10
save_interval=100
dry_run=false
while (($#)); do
    case "$1" in
        --max_iterations|--num_envs|--seed|--save_interval)
            option="$1"
            (($# >= 2)) || fail "$option requires a value"
            value="$2"
            if [[ "$option" == --seed ]]; then
                [[ "$value" =~ ^[0-9]+$ ]] || fail "$option must be a nonnegative integer"
            else
                [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "$option must be a positive integer"
            fi
            case "$option" in
                --max_iterations) iterations="$value" ;;
                --num_envs) num_envs="$value" ;;
                --seed) seed="$value" ;;
                --save_interval) save_interval="$value" ;;
            esac
            shift 2
            ;;
        --headless) shift ;;  # All six child commands already include --headless.
        --dry-run) dry_run=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) fail "Unknown option: $1 (use --help)" ;;
    esac
done

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_bin="$project_root/.venv/bin/python"
if [[ "$dry_run" == false ]]; then
    source "$project_root/scripts/activate_train.sh"
    python_bin="$VIRTUAL_ENV/bin/python"
fi
trap 'printf "\nBatch interrupted; remaining experiments were not started.\n" >&2; exit 130' INT

tasks=(
    exp451_baseline
    exp451_critic_truth
    exp451_actor_critic_truth
    exp451_est_128_64
    exp451_est_64_64
    exp451_est_64_32
)
for i in "${!tasks[@]}"; do
    task="${tasks[$i]}"
    command=("$python_bin" -u wheel_legged_gym/scripts/train.py
             --task "$task" --headless --seed "$seed" --num_envs "$num_envs"
             --max_iterations "$iterations" --save_interval "$save_interval")
    printf '\n[%d/%d] %s\n' "$((i + 1))" "${#tasks[@]}" "$task"
    if [[ "$dry_run" == true ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
    elif "${command[@]}"; then
        printf '[%d/%d] Completed: %s\n' "$((i + 1))" "${#tasks[@]}" "$task"
    else
        status=$?
        printf 'Stopped: %s exited with code %d. Remaining experiments were not started.\n' "$task" "$status" >&2
        exit "$status"
    fi
done
if [[ "$dry_run" == false ]]; then
    printf '\nAll six experiments completed. Results: %s/logs/chapter4_5_1/\n' "$project_root"
fi
