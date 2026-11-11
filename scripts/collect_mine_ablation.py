"""Run the five trained ablations with one fixed raw-data collection protocol."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wheel_legged_gym.evaluation.config import parse_args, protocol_dict, resolve_checkpoint

VARIANTS = ("full", "no_latent_input", "no_gate", "no_state_input", "no_mode_loss")


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--envs-per-terrain", type=int, default=32)
    parser.add_argument("--episodes-per-env", type=int, default=5)
    parser.add_argument("--speed", type=float, default=1.)
    parser.add_argument("--max-seconds", type=float, default=30.)
    parser.add_argument("--spawn-xy-jitter", type=float, default=0.)
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    jobs = []
    reference_cfg = None
    for variant in VARIANTS:
        task = "mine_ablation_" + variant
        common = ["--task", task, "--seed", str(args.seed),
                  "--envs-per-terrain", str(args.envs_per_terrain),
                  "--episodes-per-env", str(args.episodes_per_env),
                  "--speed", str(args.speed), "--max-seconds", str(args.max_seconds),
                  "--spawn-xy-jitter", str(args.spawn_xy_jitter),
                  "--sim-device", args.sim_device, "--headless"]
        selection = parse_args(common + ["--checkpoint", str(args.checkpoint)])
        checkpoint = resolve_checkpoint(selection, ROOT)
        snapshots = {name: json.loads((checkpoint.parent / name).read_text(encoding="utf-8"))
                     for name in ("run_info.json", "env_cfg.json", "train_cfg.json")}
        env_cfg = snapshots["env_cfg.json"]
        if reference_cfg is not None and env_cfg != reference_cfg:
            raise ValueError("Training environment configs differ: " + task)
        reference_cfg = env_cfg
        policy = snapshots["train_cfg.json"]["policy"]
        expected = {"use_latent_input": variant != "no_latent_input",
                    "use_state_input": variant != "no_state_input",
                    "use_mode_gate": variant != "no_gate",
                    "mode_loss_coef": 0. if variant in ("no_gate", "no_mode_loss") else .5}
        if any(policy.get(key) != value for key, value in expected.items()):
            raise ValueError("Checkpoint ablation switches do not match task: " + task)
        command = ["bash", str(ROOT / "scripts/evaluate_mine.sh")] + common + [
            "--checkpoint-path", str(checkpoint), "--output-dir", str(output / variant)]
        jobs.append(dict(variant=variant, task=task, status="pending", command=command,
                         checkpoint=str(checkpoint),
                         checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                         training_seed=snapshots["run_info.json"]["seed"],
                         policy_switches=expected, protocol=protocol_dict(selection)))
    manifest = dict(status="planned", created_utc=now(), jobs=jobs,
                    total_attempts=sum(job["protocol"]["total_episodes"] for job in jobs),
                    data_mode="raw_formula_inputs", metrics_computed=False)
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite existing collection: " + str(output))
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "collection_manifest.json"
    manifest["status"] = "running"
    save(manifest_path, manifest)
    try:
        for job in jobs:
            job.update(status="running", started_utc=now())
            save(manifest_path, manifest)
            print("Collecting {}: {} attempts".format(job["variant"], job["protocol"]["total_episodes"]), flush=True)
            with (output / (job["variant"] + ".log")).open("w", encoding="utf-8") as log:
                result = subprocess.run(job["command"], cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
            job.update(returncode=result.returncode, finished_utc=now())
            if result.returncode:
                job["status"] = "error"
                raise RuntimeError("Collection failed; see " + str(output / (job["variant"] + ".log")))
            metadata = json.loads((output / job["variant"] / "metadata.json").read_text())
            if metadata["status"] != "complete" or metadata["recorded_episodes"] != job["protocol"]["total_episodes"]:
                raise RuntimeError("Incomplete collection: " + job["variant"])
            job["status"] = "complete"
            job["recorded_episodes"] = metadata["recorded_episodes"]
            save(manifest_path, manifest)
            print("Completed " + job["variant"], flush=True)
        counts = []
        for job in jobs:
            with (output / job["variant"] / "summary.csv").open(newline="") as handle:
                counts.extend(dict(variant=job["variant"], training_seed=job["training_seed"], **row)
                              for row in csv.DictReader(handle))
        with (output / "event_counts.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(counts[0]))
            writer.writeheader()
            writer.writerows(counts)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error", error=repr(exc))
        raise
    finally:
        manifest["finished_utc"] = now()
        save(manifest_path, manifest)
    print("Raw data collection complete: " + str(output), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
