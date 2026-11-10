"""Checkpoint/task safety and reproducible evaluation configuration."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import isaacgym  # Must precede torch when running alongside policy tests.
from wheel_legged_gym import envs
from wheel_legged_gym.utils import task_registry
from wheel_legged_gym.utils.helpers import class_to_dict
from wheel_legged_gym.evaluation.config import (
    apply_snapshot, configure_evaluation, load_run_config, parse_args,
    protocol_dict, resolve_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]


class EvaluationConfigTests(unittest.TestCase):
    def test_local_patch_is_default_and_uses_the_entire_crossing_window(self):
        args = parse_args([])
        self.assertEqual(args.layout, "patch")
        self.assertEqual(args.size_m, 6.2)
        self.assertFalse(hasattr(args, "warmup_seconds"))
        self.assertFalse(hasattr(args, "vir_threshold"))
        self.assertAlmostEqual(args.max_seconds, 20.4)
        self.assertEqual(args.wheel_side, "all")
        self.assertEqual(args.terrain_types, ("flat", "stairs_concave_15cm", "stairs_concave_20cm",
                         "stairs_convex_15cm", "stairs_convex_20cm", "rough", "slope"))
        self.assertEqual(args.rough_height, .04)
        self.assertEqual(args.slope_angle_deg, 12.)
        self.assertEqual(args.step_width, .2)
        protocol = protocol_dict(args)
        self.assertIn("last of all four wheels", protocol["measurement_window"])
        self.assertFalse(protocol["entry_requires_contact"])
        self.assertEqual(protocol["data_mode"], "raw_formula_inputs")
        self.assertEqual(protocol["total_episodes"], 112)
        self.assertEqual(protocol["version"], 6)
        self.assertEqual(protocol["stair_geometry"]["steps_per_side"], 8)
        self.assertEqual(protocol["stair_geometry"]["tread_width_m"], .2)
        self.assertEqual(protocol["stair_geometry"]["central_platform_size_m"], 3.)
        self.assertEqual(protocol["stair_geometry"]["center_depth_or_height_m"], [1.2, 1.6])
        self.assertEqual(protocol["required_wheels"], ["FL", "FR", "RL", "RR"])
        self.assertIn("all four wheels entered", protocol["success_rule"])
        self.assertNotIn("step_height", protocol["terrain_parameters"])
        self.assertNotIn("obstacle_height", protocol["terrain_parameters"])
        self.assertNotIn("vir_threshold_mps2", protocol)
        reverse = parse_args(["--speed", "-1"])
        self.assertAlmostEqual(reverse.max_seconds, 20.4)
        self.assertAlmostEqual(parse_args(["--speed", ".5"]).max_seconds, 35.8)
        self.assertEqual(parse_args(["--max-seconds", "45"]).max_seconds, 45.)
        self.assertEqual(protocol_dict(reverse)["entry_edge"], "xmax")
        self.assertEqual(protocol_dict(reverse)["travel_direction"], "-x")
        for invalid in (["--size-m", "40"], ["--speed", "0"],
                        ["--spawn-xy-jitter", ".1"], ["--size-m", "3"],
                        ["--terrain-types", "discrete_obstacles"],
                        ["--terrain-types", "stairs_up"], ["--wheel-side", "left"],
                        ["--step-height", ".2"], ["--step-width", "1"], ["--step-width", ".5"],
                        ["--vir-threshold", "2"], ["--warmup-seconds", "1"],
                        ["--trajectory-stride", "0"], ["--trajectory-stride", "2"]):
            with self.subTest(invalid=invalid), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(invalid)

    def test_local_patch_uses_actual_asset_geometry_for_wheel_alignment(self):
        cfg, _ = task_registry.get_cfgs("mine_ablation_full")
        before = class_to_dict(cfg)
        actual = configure_evaluation(cfg, parse_args(["--speed", "-1"]))
        self.assertEqual(class_to_dict(cfg), before)
        self.assertEqual(actual.terrain.terrain_length, 6.2)
        self.assertEqual(actual.terrain.terrain_width, 6.2)
        self.assertEqual(actual.terrain.horizontal_scale, .02)
        self.assertEqual(actual.terrain.corridor_width, 8.2)
        self.assertEqual(actual.evaluation.wheel_side, "all")
        self.assertLess(actual.terrain.wheel_y_min, actual.terrain.right_wheel_y)
        self.assertGreater(actual.terrain.wheel_y_max, actual.terrain.left_wheel_y)
        self.assertAlmostEqual(actual.terrain.left_wheel_y, .236215, places=5)
        self.assertAlmostEqual(actual.terrain.right_wheel_y, -.260380, places=5)
        self.assertEqual(actual.evaluation.direction, "reverse")
        self.assertEqual(actual.evaluation.command_vx, -1)
        self.assertFalse(hasattr(actual.evaluation, "warmup_s"))
        self.assertFalse(hasattr(actual.evaluation, "vir_threshold_mps2"))
        self.assertEqual(actual.evaluation.wheel_geometry["wheel_names"],
                         ["FL_foot", "FR_foot", "RL_foot", "RR_foot"])

    def test_explicit_units_and_route_based_timeout(self):
        args = parse_args(["--layout", "region", "--size-m", "40"])
        self.assertEqual(args.max_seconds, 82.)
        self.assertEqual(protocol_dict(args)["total_episodes"], 80)
        self.assertNotIn("discrete_obstacles", args.terrain_types)
        for invalid in ([], ["--size-m", ".4"], ["--size-m", "3.2"],
                        ["--size-m", "4", "--spawn-inset", "1.5"],
                        ["--size-m", "4", "--max-seconds", "nan"],
                        ["--size-m", "4", "--terrain-types", "discrete_obstacles"]):
            with self.subTest(invalid=invalid), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--layout", "region"] + invalid)

    def test_checkpoint_filter_and_numeric_iteration_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for run, task, models in (("001", "mine_ablation_full", [20, 100]),
                                      ("002", "mine_ablation_no_gate", [200])):
                folder = root / run
                folder.mkdir()
                (folder / "run_info.json").write_text(json.dumps({"task": task}))
                for index in models:
                    (folder / ("model_%d.pt" % index)).touch()
            args = parse_args(["--log-root", tmp])
            self.assertEqual(resolve_checkpoint(args, ROOT), root / "001/model_100.pt")
            args.load_run = "002"
            with self.assertRaisesRegex(ValueError, "different task"):
                resolve_checkpoint(args, ROOT)

    def test_explicit_checkpoint_task_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model_0.pt"
            model.touch()
            model.with_name("run_info.json").write_text('{"task":"mine_ablation_no_gate"}')
            args = parse_args(["--checkpoint-path", str(model)])
            with self.assertRaises(ValueError):
                resolve_checkpoint(args, ROOT)

    def test_snapshot_roundtrip_keeps_joint_maps_and_saved_policy(self):
        cfg, train = task_registry.get_cfgs("mine_ablation_full")
        original = class_to_dict(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "run_info.json").write_text('{"task":"mine_ablation_full","seed":23}')
            saved_train = class_to_dict(train)
            saved_train["policy"]["actor_hidden_dims"] = [128, 64]
            for name, values in (("env_cfg.json", original), ("train_cfg.json", saved_train)):
                (folder / name).write_text(json.dumps(values))
            result = load_run_config(cfg, train, folder / "model_1.pt", "mine_ablation_full")
        self.assertEqual(result["source"], "training_run_snapshots")
        self.assertEqual(class_to_dict(cfg), original)
        self.assertIsInstance(cfg.init_state.default_joint_angles, dict)
        self.assertIsInstance(cfg.control.stiffness, dict)
        self.assertEqual(train.policy.actor_hidden_dims, [128, 64])
        holder = SimpleNamespace()
        apply_snapshot(holder, {"new_mapping": {"joint": 1.}})
        self.assertEqual(holder.new_mapping["joint"], 1.)

    def test_legacy_config_needs_explicit_opt_in(self):
        cfg, train = task_registry.get_cfgs("mine_ablation_full")
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "model_0.pt"
            with self.assertRaisesRegex(ValueError, "allow-current-config"):
                load_run_config(cfg, train, checkpoint, "mine_ablation_full")
            result = load_run_config(cfg, train, checkpoint, "mine_ablation_full", allow_current=True)
            self.assertEqual(result["source"], "current_registered_config")

    def test_fixed_protocol_preserves_training_dynamics_and_configs(self):
        cfg, _ = task_registry.get_cfgs("mine_ablation_full")
        before = class_to_dict(cfg)
        args = parse_args(["--envs-per-terrain", "2", "--episodes-per-env", "3"])
        actual = configure_evaluation(cfg, args, Path("model_20.pt"))
        self.assertEqual(class_to_dict(cfg), before)
        self.assertEqual(class_to_dict(actual.control), before["control"])
        self.assertEqual(class_to_dict(actual.asset), before["asset"])
        self.assertEqual(actual.sim.dt, cfg.sim.dt)
        self.assertEqual(actual.env.num_observations, cfg.env.num_observations)
        self.assertEqual(actual.env.num_envs, 14)
        self.assertEqual(actual.evaluation.episodes_per_env, 3)
        self.assertFalse(actual.commands.curriculum)
        self.assertFalse(actual.terrain.curriculum)
        self.assertFalse(actual.noise.add_noise)
        self.assertFalse(actual.domain_rand.randomize_torque_scale)
        for name in dir(actual.domain_rand):
            value = getattr(actual.domain_rand, name)
            if isinstance(value, bool):
                self.assertFalse(value, name)

    def test_protocol_identical_across_ablation_tasks(self):
        reference = None
        for variant in ("full", "no_latent_input", "no_gate", "no_state_input", "no_mode_loss"):
            task = "mine_ablation_" + variant
            cfg, _ = task_registry.get_cfgs(task)
            args = parse_args(["--task", task])
            actual = class_to_dict(configure_evaluation(cfg, args))
            actual["evaluation"].pop("task")
            if reference is None:
                reference = actual
            self.assertEqual(actual, reference)

    def test_dry_run_needs_no_checkpoint_or_simulator(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "should_not_exist"
            result = subprocess.run([sys.executable, "-m", "wheel_legged_gym.scripts.evaluate",
                                     "--size-m", "6.2", "--dry-run", "--log-root", tmp,
                                     "--output-dir", str(output)], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            values = json.loads(result.stdout)
            self.assertTrue(values["dry_run"])
            self.assertEqual(values["layout"], "patch")
            self.assertEqual(values["size_m"], 6.2)
            self.assertEqual(values["total_episodes"], 112)
            self.assertIsNone(values["checkpoint"])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
