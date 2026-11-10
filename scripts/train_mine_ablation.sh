#!/usr/bin/env bash
# Run each ablation with the same ordered set of random seeds.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash scripts/train_mine_ablation.sh [options]

  --max_iterations N  Iterations per run (default: 10000)
  --num_envs N        Parallel environments within a run (default: 4096)
  --save_interval N   Checkpoint interval (default: 200)
  --seeds N,N,...     Ordered nonnegative seeds (default: 10,20,30)
  --seed N            Use one nonnegative seed instead of --seeds
  --headless          Disable the viewer (already enabled by default)
  --dry-run           Print commands without activating Python or training
  --help, -h          Show this help

Order: full, no_latent_input, no_gate, no_state_input, no_mode_loss;
all specified seeds run for one variant before the next variant starts.
A failed run or Ctrl+C stops the batch and preserves existing results.
Task names, experiment name and run names are fixed by this script.
HELP
}

fail() {
    printf 'Error: %s\n' "$1" >&2
    exit 2
}

iterations=10000
num_envs=4096
save_interval=200
seed_values=10,20,30
seed_option=
dry_run=false
while (($#)); do
    case "$1" in
        --max_iterations|--num_envs|--save_interval|--seeds|--seed)
            option="$1"
            (($# >= 2)) || fail "$option requires a value"
            value="$2"
            case "$option" in
                --seeds|--seed)
                    [[ -z "$seed_option" || "$seed_option" == "$option" ]] || fail "Use either --seed or --seeds, not both"
                    if [[ "$option" == --seed ]]; then
                        [[ "$value" =~ ^(0|[1-9][0-9]*)$ ]] || fail "--seed must be a nonnegative integer"
                    else
                        [[ "$value" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]] || fail "--seeds must contain comma-separated nonnegative integers"
                    fi
                    seed_option="$option"
                    seed_values="$value"
                    ;;
                *)
                    [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "$option must be a positive integer"
                    case "$option" in
                        --max_iterations) iterations="$value" ;;
                        --num_envs) num_envs="$value" ;;
                        --save_interval) save_interval="$value" ;;
                    esac
                    ;;
            esac
            shift 2
            ;;
        --headless) shift ;;
        --dry-run) dry_run=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) fail "Unsupported option: $1 (task and output grouping cannot be overridden; use --help)" ;;
    esac
done

IFS=, read -r -a seeds <<< "$seed_values"
declare -A seen_seeds=()
for seed in "${seeds[@]}"; do
    [[ -z "${seen_seeds[$seed]+present}" ]] || fail "Duplicate seed: $seed"
    seen_seeds[$seed]=1
done

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_bin="$project_root/.venv/bin/python"
if [[ "$dry_run" == false ]]; then
    source "$project_root/scripts/activate_train.sh"
    python_bin="$VIRTUAL_ENV/bin/python"
fi
trap 'printf "\nBatch interrupted; remaining runs were not started.\n" >&2; exit 130' INT

variants=(full no_latent_input no_gate no_state_input no_mode_loss)
total_runs=$((${#variants[@]} * ${#seeds[@]}))
run_number=0
for variant in "${variants[@]}"; do
    for seed in "${seeds[@]}"; do
        run_number=$((run_number + 1))
        task="mine_ablation_$variant"
        run_name="${variant}_seed${seed}"
        command=("$python_bin" -u wheel_legged_gym/scripts/train.py
                 --task "$task" --headless --seed "$seed" --num_envs "$num_envs"
                 --max_iterations "$iterations" --save_interval "$save_interval"
                 --experiment_name mine_ablation --run_name "$run_name")
        printf '\n[%d/%d] %s\n' "$run_number" "$total_runs" "$run_name"
        if [[ "$dry_run" == true ]]; then
            printf '%q ' "${command[@]}"
            printf '\n'
        elif "${command[@]}"; then
            printf '[%d/%d] Completed: %s\n' "$run_number" "$total_runs" "$run_name"
        else
            status=$?
            printf 'Stopped: %s exited with code %d. Remaining runs were not started.\n' "$run_name" "$status" >&2
            exit "$status"
        fi
    done
done
if [[ "$dry_run" == false ]]; then
    printf '\nAll %d runs completed. Results: %s/logs/mine_ablation/\n' "$total_runs" "$project_root"
fi
