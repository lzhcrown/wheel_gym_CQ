"""Fixed-terrain parallel raw-data recording, separate from training reward CSV."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from wheel_legged_gym.evaluation.config import (
    configure_evaluation, load_run_config, parse_args, protocol_dict, resolve_checkpoint,
)


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_metadata(output, metadata):
    temporary = output / "metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "metadata.json")


def evaluate(args):
    protocol = protocol_dict(args)
    try:
        checkpoint = resolve_checkpoint(args, ROOT)
    except FileNotFoundError as exc:
        if not args.dry_run or args.checkpoint_path is not None or args.load_run != "-1" or args.checkpoint != -1:
            raise
        checkpoint = None
        protocol["checkpoint_selection"] = str(exc)
    if args.dry_run:
        protocol["checkpoint"] = str(checkpoint) if checkpoint else None
        protocol["dry_run"] = True
        print(json.dumps(protocol, ensure_ascii=False, indent=2))
        return 0

    # Import ordering is required by the Isaac Gym Python binding. --help and
    # --dry-run above deliberately require neither Gym nor Torch.
    import isaacgym
    from isaacgym import gymapi
    import torch
    from wheel_legged_gym import envs
    from wheel_legged_gym.utils import task_registry
    from wheel_legged_gym.utils.helpers import class_to_dict, parse_sim_params, set_seed
    from wheel_legged_gym.algo.Mine.actor_critic_mine import MINEActorCritic
    from wheel_legged_gym.evaluation.env import EvaluationEnv
    from wheel_legged_gym.evaluation.patch_env import PatchEvaluationEnv
    from wheel_legged_gym.evaluation.metrics import CsvEvaluationWriter

    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    provenance = load_run_config(env_cfg, train_cfg, checkpoint, args.task, args.allow_current_config)
    if train_cfg.runner_class_name != "MINEOnPolicyRunner" or train_cfg.runner.policy_class_name != "MINEActorCritic":
        raise ValueError("Evaluation requires a MINE checkpoint/config")
    if provenance["source"] == "current_registered_config":
        print("Using explicitly requested current configs for a legacy model; check ablation switches and observation definitions.")
    cfg = configure_evaluation(env_cfg, args, checkpoint)
    if args.sim_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; evaluation has not started. Check the NVIDIA driver or explicitly choose --sim-device cpu for a small smoke test.")
    set_seed(args.seed)
    model = MINEActorCritic(
        cfg.env.num_observations, cfg.env.num_privileged_obs,
        cfg.env.num_single_obs, cfg.env.single_num_privileged_obs,
        cfg.env.num_est_prob, cfg.env.num_actions, **class_to_dict(train_cfg.policy),
    ).to(args.sim_device)
    saved = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.eval()
    del saved
    sim_args = SimpleNamespace(physics_engine=gymapi.SIM_PHYSX, device=args.sim_device,
                               use_gpu=args.sim_device.startswith("cuda"), subscenes=0,
                               use_gpu_pipeline=args.sim_device.startswith("cuda"), num_threads=args.num_threads)
    sim_params = parse_sim_params(sim_args, {"sim": class_to_dict(cfg.sim)})
    output = args.output_dir or ROOT / "evaluations" / args.task / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new or empty evaluation output directory: " + str(output))
    output.mkdir(parents=True, exist_ok=True)
    asset = Path(cfg.asset.file.format(WHEEL_LEGGED_GYM_ROOT_DIR=str(ROOT)))
    metadata = {
        "status": "initializing", "started_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "config_provenance": provenance, "protocol": protocol,
        "env_cfg": class_to_dict(cfg), "policy_cfg": class_to_dict(train_cfg.policy),
        "robot_asset": str(asset), "robot_asset_sha256": sha256(asset),
        "physics_dt_s": sim_params.dt, "policy_dt_s": sim_params.dt * cfg.control.decimation,
        "data_mode": "raw_formula_inputs", "raw_schema_version": 1,
        "units": {"position": "m", "linear_velocity": "m/s", "linear_acceleration": "m/s^2",
                  "angular_velocity": "rad/s", "joint_torque": "N*m", "time": "s", "mass": "kg"},
        "gravity_vector_m_s2": [sim_params.gravity.x, sim_params.gravity.y, sim_params.gravity.z],
        "sim_device": args.sim_device,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (
            Path(__file__).resolve(), ROOT / "wheel_legged_gym/evaluation/config.py",
            ROOT / "wheel_legged_gym/evaluation/env.py", ROOT / "wheel_legged_gym/evaluation/terrain.py",
            ROOT / "wheel_legged_gym/evaluation/metrics.py",
            ROOT / "wheel_legged_gym/evaluation/patch_env.py",
            ROOT / "wheel_legged_gym/evaluation/patch_terrain.py",
            ROOT / "wheel_legged_gym/evaluation/presets.py",
            ROOT / "wheel_legged_gym/evaluation/crossing.py",
            ROOT / "wheel_legged_gym/evaluation/geometry.py",
            ROOT / "wheel_legged_gym/algo/Mine/actor_critic_mine.py",
            ROOT / "wheel_legged_gym/algo/Mine/estimator_mine.py",
            ROOT / "wheel_legged_gym/envs/nezha2/legged_robot.py",
            ROOT / "wheel_legged_gym/envs/nezha2/nezha2_env.py",
        )},
    }
    write_metadata(output, metadata)
    print("Checkpoint: {}\nEvaluation CSV: {}".format(checkpoint, output))
    env = None
    status, reason = "error", "evaluation_error"
    with CsvEvaluationWriter(output, num_joints=cfg.env.num_actions, save_trajectories=True) as writer:
        try:
            environment_class = PatchEvaluationEnv if args.layout == "patch" else EvaluationEnv
            env = environment_class(cfg, sim_params, gymapi.SIM_PHYSX, args.sim_device, args.headless,
                                    episode_sink=writer.write_episodes, trajectory_sink=writer.write_trajectories,
                                    physics_sink=writer.write_physics)
            if env.num_dof != cfg.env.num_actions or len(env.measurement_joint_names) != env.num_dof:
                raise ValueError("Raw joint-column mapping does not match the simulated DOF count")
            metadata["joint_names"] = list(env.measurement_joint_names)
            metadata["joint_columns"] = [{
                "index": i, "name": name,
                "torque": "torque_j%02d_nm" % i,
                "velocity_before": "joint_vel_before_j%02d_radps" % i,
                "velocity_after": "joint_vel_after_j%02d_radps" % i,
            } for i, name in enumerate(env.measurement_joint_names)]
            metadata["sampling"] = {
                "physics_hz": 1. / sim_params.dt,
                "control_hz": 1. / env.dt,
                "acceleration_definition": "(current world XYZ velocity - previous world XYZ velocity) / control_dt_s",
                "quaternion_order": "xyzw",
                "command_frame": "body",
                "analysis_threshold_or_warmup_applied": False,
                "window_resolution_s": env.dt,
            }
            metadata["terrain"] = env.terrain.metadata
            metadata["total_mass_kg"] = env.total_mass_kg.cpu().tolist()
            metadata["status"] = "running"
            write_metadata(output, metadata)
            env.reset()
            if not args.headless:
                center = env.spawn_positions[0].cpu().tolist()
                env.set_camera([center[0] - 3, center[1] - 3, center[2] + 2], center)
            last_report = time.monotonic()
            with torch.inference_mode():
                while not env.evaluation_complete:
                    actions = model.act_inference(env.get_observations())
                    env.step(actions)
                    if time.monotonic() - last_report >= 10:
                        print("Completed {}/{} traversal attempts".format(len(env.completed_episodes), protocol["total_episodes"]), flush=True)
                        last_report = time.monotonic()
            status, reason = "complete", ""
        except (KeyboardInterrupt, SystemExit):
            status, reason = "interrupted", "user_interrupt_or_viewer_closed"
        except Exception as exc:
            metadata["error"] = repr(exc)
            raise
        finally:
            try:
                if env is not None and status != "complete":
                    env.interrupt(reason=reason)
                rows = env.completed_episodes if env is not None else []
                summary = writer.write_summary(rows)
                metadata["status"] = status
                metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
                metadata["recorded_episodes"] = len(rows)
                write_metadata(output, metadata)
            finally:
                if env is not None:
                    if env.viewer is not None:
                        env.gym.destroy_viewer(env.viewer)
                    env.gym.destroy_sim(env.sim)
    for row in summary:
        if row["scope"] == "terrain":
            print("{}: success={}, completed attempts={}, interrupted={}".format(row["terrain"], row["success"], row["completed_episodes"], row["interrupted"]))
    print("Evaluation {}: {}".format(status, output))
    return 0 if status == "complete" else 130


if __name__ == "__main__":
    try:
        sys.exit(evaluate(parse_args()))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print("Evaluation error: " + str(exc), file=sys.stderr)
        sys.exit(1)
