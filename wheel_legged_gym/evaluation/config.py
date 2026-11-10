"""Explicit evaluation protocol and checkpoint selection (no simulator imports)."""
import argparse
import copy
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace

from .presets import (PATCH_TERRAIN_TYPES, PATCH_SIZE_M,
                      PATCH_PLATFORM_LENGTH_M, PATCH_CORRIDOR_WIDTH_M,
                      PATCH_STAIR_TREAD_M, PATCH_STEPS_PER_SIDE, PATCH_STAIR_PLATFORM_M)

TERRAINS = PATCH_TERRAIN_TYPES
REGION_TERRAINS = ("flat", "rough", "slope", "stairs_up", "stairs_down")
TASKS = ("nezha2_mine",) + tuple("mine_ablation_" + name for name in (
    "full", "no_latent_input", "no_gate", "no_state_input", "no_mode_loss"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run trained MINE policies on fixed terrain and record raw formula inputs to CSV for offline analysis.")
    parser.add_argument("--task", choices=TASKS, default="mine_ablation_full")
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--load_run", "--load-run", default="-1")
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--allow-current-config", action="store_true",
                        help="Explicitly use current task configs for a legacy run without complete config snapshots")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--layout", choices=("patch", "region"), default="patch",
                        help="patch: wheel-entry/last-wheel-exit window around a local square; region: legacy large-region evaluation")
    parser.add_argument("--size-m", type=float,
                        help="Test area side length in METRES (default {:g}); legacy region requires an explicit size".format(PATCH_SIZE_M))
    parser.add_argument("--terrain-types", help="Comma-separated terrain names; patch defaults to all seven equally sized presets")
    parser.add_argument("--envs-per-terrain", type=int, default=16)
    parser.add_argument("--episodes-per-env", "--attempts-per-env", type=int, default=1,
                        help="Traversal attempts per parallel slot; failures before entering the patch are reported separately")
    parser.add_argument("--seed", type=int, default=10, help="Evaluation seed, independent of training seed")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Constant body-frame vx, m/s; negative values evaluate rear-first reverse travel in patch mode")
    parser.add_argument("--max-seconds", type=float,
                        help="Whole attempt limit including approach: default twice nominal platform-to-platform travel time plus 5 s")
    parser.add_argument("--trajectory-stride", type=int, choices=(1,), default=1,
                        help="Raw capture requires every control step (1); all physics substeps are also recorded")
    parser.add_argument("--spawn-inset", type=float, default=0.75)
    parser.add_argument("--goal-inset", type=float, default=0.75)
    parser.add_argument("--lateral-margin", type=float, default=0.75)
    parser.add_argument("--wheel-side", choices=("all",), default="all",
                        help="All four wheels must enter and cross the test area")
    parser.add_argument("--spawn-xy-jitter", type=float, default=.04,
                        help="Initial XY jitter in metres around the centered approach")
    parser.add_argument("--max-tilt-deg", type=float, default=60.0)
    parser.add_argument("--slope-angle-deg", type=float, default=12.0)
    parser.add_argument("--rough-height", type=float, default=0.04)
    parser.add_argument("--step-height", type=float,
                        help="Legacy region only; patch variants fix each rise to 0.15 or 0.20 m")
    parser.add_argument("--step-width", type=float,
                        help="Tread length: patch fixed {:g} m ({} concentric levels), legacy region default .4 m".format(PATCH_STAIR_TREAD_M, PATCH_STEPS_PER_SIDE))
    parser.add_argument("--sim-device", default="cuda:0", help="cuda:N or cpu")
    parser.add_argument("--num-threads", type=int, default=10)
    view = parser.add_mutually_exclusive_group()
    view.add_argument("--headless", action="store_true", help="Default: run without viewer")
    view.add_argument("--viewer", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print protocol and, if present, selected checkpoint; create no simulation or files")
    args = parser.parse_args(argv)
    if args.size_m is None:
        if args.layout == "region":
            parser.error("--layout region requires an explicit --size-m in metres")
        args.size_m = PATCH_SIZE_M
    if args.step_width is None:
        args.step_width = PATCH_STAIR_TREAD_M if args.layout == "patch" else .4
    if args.layout == "patch" and args.step_height is not None:
        parser.error("Patch stair heights are fixed by their 15cm/20cm terrain names; --step-height is legacy region only")
    if args.step_height is None:
        args.step_height = .1
    allowed_terrains = TERRAINS if args.layout == "patch" else REGION_TERRAINS
    if args.terrain_types is None:
        args.terrain_types = ",".join(allowed_terrains)
    args.terrain_types = tuple(value.strip() for value in args.terrain_types.split(","))
    if not args.terrain_types or len(set(args.terrain_types)) != len(args.terrain_types) or set(args.terrain_types) - set(allowed_terrains):
        parser.error("--terrain-types must contain unique names from " + ",".join(allowed_terrains))
    for name in ("size_m", "spawn_inset", "goal_inset", "lateral_margin", "step_width", "max_tilt_deg"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error("--" + name.replace("_", "-") + " must be finite and positive")
    for name in ("slope_angle_deg", "rough_height", "step_height", "step_width", "spawn_xy_jitter"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error("--" + name.replace("_", "-") + " must be finite and nonnegative")
    if not math.isfinite(args.speed) or args.speed == 0 or (args.layout == "region" and args.speed < 0):
        parser.error("--speed must be finite and nonzero; legacy region mode requires positive speed")
    grid = .02 if args.layout == "patch" else .1
    if not math.isclose(args.size_m / grid, round(args.size_m / grid), abs_tol=1e-6):
        parser.error("--size-m must be a multiple of the %.2f m terrain grid" % grid)
    if args.layout == "patch":
        if not math.isclose(args.size_m, PATCH_SIZE_M, rel_tol=0., abs_tol=1e-9):
            parser.error("This evaluation protocol uses an exact {0:g} x {0:g} m test area".format(PATCH_SIZE_M))
        if args.spawn_xy_jitter > .05:
            parser.error("--spawn-xy-jitter must be <= .05 m for the external start platform")
        if not math.isclose(args.step_width, PATCH_STAIR_TREAD_M, rel_tol=0., abs_tol=1e-9):
            parser.error("The stair presets require --step-width {:g}: {} levels and a {:g} m center platform".format(
                PATCH_STAIR_TREAD_M, PATCH_STEPS_PER_SIDE, PATCH_STAIR_PLATFORM_M))
    else:
        if args.size_m < 3.2:
            parser.error("Legacy region requires at least 3.2 m for its two internal 1.5 m platforms; the default patch layout keeps platforms outside the test area")
        if args.spawn_inset >= 1.5 or args.goal_inset >= 1.5 or 2 * args.lateral_margin >= args.size_m:
            parser.error("spawn/goal insets must lie within the 1.5 m platforms; lateral margins must leave a nonempty interval")
    if args.max_tilt_deg >= 90 or args.slope_angle_deg >= 90:
        parser.error("tilt/slope angles must be less than 90 degrees")
    if args.step_width < grid:
        parser.error("--step-width must be at least the terrain grid spacing")
    interior = args.size_m - (2 * grid if args.layout == "patch" else 3.)
    if any(name.startswith("stairs_") for name in args.terrain_types) and args.step_width > interior:
        parser.error("The terrain interior must fit at least one complete stair tread")
    if min(args.envs_per_terrain, args.episodes_per_env) < 1 or min(args.seed, args.trajectory_stride, args.num_threads) < 0:
        parser.error("environment/episode counts must be positive; seed, stride and thread count must be nonnegative")
    if args.max_seconds is None:
        route_length = (args.size_m + PATCH_PLATFORM_LENGTH_M if args.layout == "patch"
                        else args.size_m - args.spawn_inset - args.goal_inset)
        args.max_seconds = 2 * route_length / abs(args.speed) + 5
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        parser.error("--max-seconds must be finite and positive")
    if args.checkpoint < -1:
        parser.error("--checkpoint must be -1 (latest) or a nonnegative iteration")
    if not re.fullmatch(r"cpu|cuda:\d+", args.sim_device):
        parser.error("--sim-device must be cpu or cuda:N")
    if args.checkpoint_path is not None and (args.log_root is not None or args.load_run != "-1" or args.checkpoint != -1):
        parser.error("use --checkpoint-path OR --log-root/--load_run/--checkpoint")
    args.headless = not args.viewer
    return args


def protocol_dict(args):
    patch = args.layout == "patch"
    return {
        "version": 6, "data_mode": "raw_formula_inputs", "layout": args.layout, "task": args.task, "size_m": args.size_m,
        "terrain_types": list(args.terrain_types), "eval_seed": args.seed,
        "envs_per_terrain": args.envs_per_terrain, "episodes_per_env": args.episodes_per_env,
        "total_episodes": len(args.terrain_types) * args.envs_per_terrain * args.episodes_per_env,
        "command_body_vx_vy_wz": [args.speed, 0., 0.], "max_seconds": args.max_seconds,
        "entry_edge": "xmin" if args.speed > 0 else "xmax", "travel_direction": "+x" if args.speed > 0 else "-x",
        "spawn_inset_m": args.spawn_inset if not patch else None,
        "goal_inset_m": args.goal_inset if not patch else None, "lateral_margin_m": args.lateral_margin if not patch else None,
        "wheel_side": args.wheel_side if patch else None,
        "spawn_xy_jitter_m": args.spawn_xy_jitter if patch else None,
        "measurement_window": "first wheel projection enters the patch until the last of all four wheels fully clears the opposite edge; keep measuring between axles" if patch else "initialization to terminal state",
        "event_sampling": "policy-step boundary; each detected boundary may lag the geometric event by up to one policy interval" if patch else "policy steps",
        "entry_requires_contact": False if patch else None,
        "success_rule": "all four wheels entered through the entry edge and all four trailing wheel edges clear the opposite edge, without failure" if patch else "base x reaches xmax - goal_inset, with y inside region and no failure",
        "required_wheels": ["FL", "FR", "RL", "RR"] if patch else None,
        "failure_rule": "persistent training contact failure, tilt limit, corridor exit, bypass or retreat; failure takes priority over success" if patch else "persistent training contact failure, tilt over limit or base exits region; failure takes priority over success",
        "max_tilt_deg": args.max_tilt_deg,
        "velocity_acceleration": "world XYZ velocity and finite-difference acceleration, no filtering, threshold or warmup exclusion",
        "trajectory_stride": args.trajectory_stride,
        "recording": "every control interval and every physics substep inside the measurement window",
        "physics_inputs": "actual applied joint torque and before/after joint angular velocity with physical dt",
        "distance_inputs": "current/previous world XYZ position plus initial goal direction; no fixed terrain-length COT denominator",
        "online_outputs": "raw samples, event rows and event counts; no power/energy/COT/RMSE/VIR/success-rate results",
        "randomization": "disabled, including 0.8-1.1 torque scaling",
        "terrain_parameters": dict({name: getattr(args, name) for name in
                               (("slope_angle_deg", "rough_height", "step_width") if patch else
                                ("slope_angle_deg", "rough_height", "step_height", "step_width"))},
                               rough_cell_size=.08 if patch else .2),
        "stair_height_interpretation": ("per riser: 0.15/0.20 m; {} levels; center depth/height {:.2f}/{:.2f} m".format(
            PATCH_STEPS_PER_SIDE, PATCH_STEPS_PER_SIDE * .15, PATCH_STEPS_PER_SIDE * .20) if patch else "per riser"),
        "stair_geometry": {"steps_per_side": PATCH_STEPS_PER_SIDE,
                           "tread_width_m": PATCH_STAIR_TREAD_M,
                           "central_platform_size_m": PATCH_STAIR_PLATFORM_M,
                           "riser_heights_m": [.15, .20],
                           "center_depth_or_height_m": [PATCH_STEPS_PER_SIDE * .15,
                                                       PATCH_STEPS_PER_SIDE * .20]} if patch else None,
        "support_platform_length_m": PATCH_PLATFORM_LENGTH_M if patch else 1.5,
        "corridor_width_m": PATCH_CORRIDOR_WIDTH_M if patch else args.size_m,
    }


def _read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_checkpoint(args, project_root):
    """Filter automatic selections by task; never silently load another ablation."""
    if args.checkpoint_path is not None:
        path = args.checkpoint_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(str(path))
    else:
        experiment = "nezha2_mine" if args.task == "nezha2_mine" else "mine_ablation"
        root = args.log_root or Path(project_root) / "logs" / experiment
        pattern = re.compile(r"model_(\d+)\.pt$")
        def models(run):
            return [(int(match.group(1)), file) for file in run.glob("model_*.pt")
                    if file.is_file() for match in [pattern.fullmatch(file.name)] if match]
        if str(args.load_run) == "-1":
            runs = [run for run in root.iterdir() if run.is_dir()
                    and (run / "run_info.json").is_file()
                    and _read_json(run / "run_info.json").get("task") == args.task and models(run)] if root.exists() else []
            if not runs:
                raise FileNotFoundError("No checkpoint run for " + args.task + " in " + str(root))
            run = sorted(runs)[-1]
        else:
            run = root / args.load_run
        if args.checkpoint == -1:
            available = models(run)
            if not available:
                raise FileNotFoundError("No model_*.pt in " + str(run))
            path = max(available, key=lambda pair: pair[0])[1].resolve()
        else:
            path = (run / ("model_%d.pt" % args.checkpoint)).resolve()
            if not path.is_file():
                raise FileNotFoundError(str(path))
    info = path.parent / "run_info.json"
    if info.is_file() and _read_json(info).get("task") != args.task:
        raise ValueError("Selected checkpoint belongs to a different task; use its matching --task")
    return path


def apply_snapshot(obj, values):
    """Recursively update config instances without replacing nested config objects."""
    for name, value in values.items():
        old = getattr(obj, name, None)
        if isinstance(value, dict) and old is not None and hasattr(old, "__dict__"):
            apply_snapshot(old, value)
        else:
            # Dictionaries such as joint-angle and PD-gain maps must remain
            # mappings: the environment indexes them by joint name.
            setattr(obj, name, copy.deepcopy(value))


def load_run_config(env_cfg, train_cfg, checkpoint, task, allow_current=False):
    root = Path(checkpoint).parent
    paths = [root / name for name in ("run_info.json", "env_cfg.json", "train_cfg.json")]
    info = _read_json(paths[0]) if paths[0].is_file() else {}
    if info and info.get("task") != task:
        raise ValueError("Checkpoint task mismatch")
    if not all(path.is_file() for path in paths):
        if not allow_current:
            raise ValueError("Run lacks run_info.json/env_cfg.json/train_cfg.json. Use a complete training run, or explicitly pass --allow-current-config for a legacy model.")
        return {"source": "current_registered_config", "run_info": info}
    apply_snapshot(env_cfg, _read_json(paths[1]))
    apply_snapshot(train_cfg, _read_json(paths[2]))
    return {"source": "training_run_snapshots", "run_info": info}


def configure_evaluation(env_cfg, args, checkpoint=None):
    """Change evaluation protocol only; keep asset, observations and PD dynamics."""
    cfg = copy.deepcopy(env_cfg)
    cfg.seed = args.seed
    cfg.env.num_envs = len(args.terrain_types) * args.envs_per_terrain
    cfg.env.episode_length_s = args.max_seconds
    cfg.noise.add_noise = False
    cfg.init_state.rand_init_dof = False
    for name in dir(cfg.domain_rand):
        if not name.startswith("_") and isinstance(getattr(cfg.domain_rand, name), bool):
            setattr(cfg.domain_rand, name, False)
    cfg.domain_rand.randomize_torque_scale = False
    cfg.domain_rand.action_noise = 0.
    cfg.domain_rand.action_delay = 0.
    cfg.commands.curriculum = False
    cfg.commands.heading_command = False
    cfg.commands.resampling_time = args.max_seconds + 1
    cfg.commands.ranges.lin_vel_x = [args.speed, args.speed]
    cfg.commands.ranges.lin_vel_y = [0., 0.]
    cfg.commands.ranges.ang_vel_yaw = [0., 0.]
    cfg.terrain.mesh_type = "trimesh"
    cfg.terrain.layout = args.layout
    cfg.terrain.curriculum = False
    cfg.terrain.teleport_robots = False
    cfg.terrain.type_names = list(args.terrain_types)
    cfg.terrain.seed = args.seed
    cfg.terrain.terrain_length = cfg.terrain.terrain_width = args.size_m
    cfg.terrain.num_rows, cfg.terrain.num_cols = 1, len(args.terrain_types)
    cfg.terrain.max_init_terrain_level = 0
    cfg.terrain.horizontal_scale, cfg.terrain.vertical_scale = (.02 if args.layout == "patch" else .1), .005
    cfg.terrain.border_size, cfg.terrain.gap_size, cfg.terrain.platform_length = 2., 2., PATCH_PLATFORM_LENGTH_M
    cfg.terrain.corridor_width = PATCH_CORRIDOR_WIDTH_M if args.layout == "patch" else args.size_m
    cfg.terrain.slope_treshold = 0.75
    for name, value in protocol_dict(args)["terrain_parameters"].items():
        setattr(cfg.terrain, name, value)
    cfg.sim.web_vis = False
    cfg.sim.keep_default_viewer = args.viewer
    cfg.sim.use_gpu_pipeline = args.sim_device.startswith("cuda")
    cfg.sim.physx.use_gpu = args.sim_device.startswith("cuda")
    cfg.evaluation = SimpleNamespace(
        episodes_per_env=args.episodes_per_env, seed=args.seed, task=args.task,
        checkpoint=str(checkpoint) if checkpoint else "", command_vx=args.speed,
        max_tilt_deg=args.max_tilt_deg, trajectory_stride=args.trajectory_stride,
        spawn_inset=args.spawn_inset, goal_inset=args.goal_inset, lateral_margin=args.lateral_margin,
        layout=args.layout, direction="forward" if args.speed > 0 else "reverse",
        wheel_side=args.wheel_side, side=args.wheel_side,
        spawn_xy_jitter=args.spawn_xy_jitter, contact_force_threshold=1.,
    )
    if args.layout == "patch":
        from .geometry import wheel_geometry_from_cfg
        geometry = wheel_geometry_from_cfg(cfg)
        cfg.evaluation.wheel_geometry = geometry
        for name in ("left_wheel_y", "right_wheel_y", "front_wheel_x", "rear_wheel_x", "wheel_radius", "wheel_x_min", "wheel_x_max", "wheel_y_min", "wheel_y_max"):
            setattr(cfg.terrain, name, geometry[name])
        cfg.terrain.wheel_geometry_source = geometry["asset_path"]
    return cfg
