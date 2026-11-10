"""Check controlled ablations and batch orchestration without a simulator."""
import contextlib
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import isaacgym  # Isaac Gym must precede torch.
import torch
import torch.nn.functional as F

from wheel_legged_gym import envs
from wheel_legged_gym.algo.Mine.actor_critic_mine import MINEActorCritic
from wheel_legged_gym.experiments.mine_ablation import CONFIGS
from wheel_legged_gym.utils import task_registry
from wheel_legged_gym.utils.helpers import class_to_dict


torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/train_mine_ablation.sh"
VARIANTS = ("full", "no_latent_input", "no_gate", "no_state_input", "no_mode_loss")
CHANGES = {
    "full": {},
    "no_latent_input": {"use_latent_input": False},
    "no_gate": {"use_mode_gate": False, "mode_loss_coef": 0.0},
    "no_state_input": {"use_state_input": False},
    "no_mode_loss": {"mode_loss_coef": 0.0},
}


def make_model(variant="full", seed=123):
    cfg, train = task_registry.get_cfgs("mine_ablation_" + variant)
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        return MINEActorCritic(
            cfg.env.num_observations, cfg.env.num_privileged_obs,
            cfg.env.num_single_obs, cfg.env.single_num_privileged_obs,
            cfg.env.num_est_prob, cfg.env.num_actions,
            **class_to_dict(train.policy))


class MineAblationTests(unittest.TestCase):
    def test_configs_change_only_declared_switches_and_log_grouping(self):
        self.assertEqual(tuple(CONFIGS), tuple("mine_ablation_" + v for v in VARIANTS))
        base_env, base_train = task_registry.get_cfgs("nezha2_mine")
        base_env, base_train = class_to_dict(base_env), class_to_dict(base_train)
        base_class = task_registry.get_task_class("nezha2_mine")
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                task = "mine_ablation_" + variant
                cfg, train = task_registry.get_cfgs(task)
                actual = class_to_dict(train)
                self.assertIs(task_registry.get_task_class(task), base_class)
                self.assertEqual(class_to_dict(cfg), base_env)
                self.assertEqual(actual["runner"]["max_iterations"], 100000)
                self.assertEqual(actual["runner"]["csv_tags"], ["Train/mean_reward"])
                self.assertEqual(actual["runner"]["experiment_name"], "mine_ablation")
                self.assertEqual(actual["runner"]["run_name"], variant)
                expected_policy = dict(base_train["policy"], **CHANGES[variant])
                self.assertEqual(actual["policy"], expected_policy)
                actual["policy"] = base_train["policy"]
                actual["runner"]["experiment_name"] = base_train["runner"]["experiment_name"]
                actual["runner"]["run_name"] = base_train["runner"]["run_name"]
                self.assertEqual(actual, base_train)

        cfg, train = task_registry.get_cfgs("mine_ablation_full")
        cfg.env.num_envs = 3
        train.policy.enc_hidden_dims[0] = 1
        train.runner.csv_tags.append("unexpected")
        fresh_cfg, fresh_train = task_registry.get_cfgs("mine_ablation_full")
        self.assertEqual(class_to_dict(fresh_cfg), base_env)
        self.assertEqual(fresh_train.policy.enc_hidden_dims, base_train["policy"]["enc_hidden_dims"])
        self.assertEqual(fresh_train.runner.csv_tags, ["Train/mean_reward"])

    def test_all_parameters_and_rng_state_match_with_fixed_seed(self):
        full = make_model()
        reference = full.state_dict()
        reference_rng = torch.get_rng_state().clone()
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = make_model(variant)
                self.assertEqual(model.actor[0].in_features, 85)
                self.assertEqual(model.critic[0].in_features, 1125)
                self.assertEqual(model.estimator.encoder[0].in_features, 130)
                self.assertEqual(model.estimator.mode_gate[0].in_features, 130)
                self.assertEqual(model.estimator.target[0].in_features, 375)
                self.assertEqual(set(model.state_dict()), set(reference))
                for name, actual in model.state_dict().items():
                    self.assertTrue(torch.equal(actual, reference[name]), name)
                self.assertTrue(torch.equal(torch.get_rng_state(), reference_rng))

    def test_actor_masks_match_for_sampling_and_inference(self):
        history = torch.linspace(-1., 1., 8 * 130).reshape(8, 130)
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = make_model(variant)
                state, latent = model.estimator(history)
                captured = []
                handle = model.actor.register_forward_pre_hook(
                    lambda module, inputs: captured.append(inputs[0].detach().clone()))
                try:
                    model.act(history)
                    sampled_mean = model.action_mean.detach().clone()
                    inferred_mean = model.act_inference(history)
                finally:
                    handle.remove()
                self.assertEqual(len(captured), 2)
                for actual in captured:
                    self.assertEqual(actual.shape, (8, 85))
                    torch.testing.assert_close(actual[:, :65], history[:, -65:])
                    torch.testing.assert_close(actual[:, 65:69],
                        torch.zeros_like(state) if variant == "no_state_input" else state)
                    torch.testing.assert_close(actual[:, 69:],
                        torch.zeros_like(latent) if variant == "no_latent_input" else latent)
                torch.testing.assert_close(sampled_mean, inferred_mean)

    def test_auxiliary_gradients_keep_input_ablations_and_bypass_disabled_gate(self):
        torch.manual_seed(29)
        history = torch.randn(16, 130)
        current, following = torch.randn(16, 1125), torch.randn(16, 1125)
        reference_gradients = None
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                estimator = make_model(variant).estimator
                with patch.object(estimator.mode_gate, "forward", wraps=estimator.mode_gate.forward) as gate, \
                     patch("wheel_legged_gym.algo.Mine.estimator_mine.F.cross_entropy", wraps=F.cross_entropy) as mode_loss:
                    state, latent = estimator(history)
                    losses = estimator.update(history, current, following, lr=0.)
                self.assertTrue(all(torch.isfinite(torch.tensor(losses))))
                self.assertEqual(gate.call_count, 0 if variant == "no_gate" else 2)
                self.assertEqual(mode_loss.call_count, 0 if variant in ("no_gate", "no_mode_loss") else 1)
                for name, parameter in estimator.named_parameters():
                    disabled = variant == "no_gate" and (name.startswith("mode_gate.") or name == "mode_prototypes")
                    if disabled:
                        self.assertIsNone(parameter.grad, name)
                    else:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                for parameter in (estimator.encoder[-1].weight, estimator.target[0].weight,
                                  estimator.proto.weight):
                    self.assertGreater(parameter.grad.abs().sum().item(), 0.)
                gradients = {name: p.grad.detach().clone() for name, p in estimator.named_parameters()
                             if p.grad is not None}
                if variant == "full":
                    reference_gradients = gradients
                elif variant in ("no_latent_input", "no_state_input"):
                    for name in gradients:
                        torch.testing.assert_close(gradients[name], reference_gradients[name], rtol=0., atol=0.)
                if variant == "no_gate":
                    raw = estimator.encoder(history)
                    torch.testing.assert_close(state, raw[:, :4])
                    torch.testing.assert_close(latent, F.normalize(raw[:, 4:], dim=-1))
                elif variant == "no_mode_loss":
                    self.assertGreater(estimator.mode_gate[0].weight.grad.abs().sum().item(), 0.)
                    self.assertGreater(estimator.mode_prototypes.grad.abs().sum().item(), 0.)


class MineAblationBatchTests(unittest.TestCase):
    def dry_run(self, *options):
        result = subprocess.run(["bash", str(SCRIPT), "--dry-run", *options],
                                cwd=str(ROOT), text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return [shlex.split(line) for line in result.stdout.splitlines()
                if "wheel_legged_gym/scripts/train.py" in line]

    def test_dry_run_default_order_and_overrides(self):
        commands = self.dry_run()
        self.assertEqual(len(commands), 15)
        expected = [(variant, seed) for variant in VARIANTS for seed in (10, 20, 30)]
        for command, (variant, seed) in zip(commands, expected):
            values = {flag: command[command.index(flag) + 1] for flag in
                      ("--task", "--seed", "--num_envs", "--max_iterations",
                       "--save_interval", "--experiment_name", "--run_name")}
            self.assertEqual(values, {
                "--task": "mine_ablation_" + variant, "--seed": str(seed),
                "--num_envs": "4096", "--max_iterations": "10000", "--save_interval": "200",
                "--experiment_name": "mine_ablation", "--run_name": variant + "_seed" + str(seed)})
            self.assertIn("--headless", command)
        commands = self.dry_run("--seeds", "7,11", "--num_envs", "32",
                                "--max_iterations", "9", "--save_interval", "3")
        self.assertEqual(len(commands), 10)
        self.assertEqual([c[c.index("--seed") + 1] for c in commands], ["7", "11"] * 5)
        for command in commands:
            for flag, value in (("--num_envs", "32"), ("--max_iterations", "9"), ("--save_interval", "3")):
                self.assertEqual(command[command.index(flag) + 1], value)
        commands = self.dry_run("--seed", "0", "--headless")
        self.assertEqual(len(commands), 5)
        self.assertTrue(all(c[c.index("--seed") + 1] == "0" for c in commands))

    def test_batch_rejects_bad_seeds_and_grouping_overrides(self):
        invalid = (("--seed", "-1"), ("--seeds", "10,10"), ("--seeds", "10,,20"),
                   ("--seed", "10", "--seeds", "20,30"), ("--max_iterations", "0"),
                   ("--task", "nezha2_mine"), ("--experiment_name", "wrong"),
                   ("--run_name", "wrong"))
        for options in invalid:
            with self.subTest(options=options):
                result = subprocess.run(["bash", str(SCRIPT), "--dry-run", *options],
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 2)
                self.assertNotIn("wheel_legged_gym/scripts/train.py", result.stdout)

    def test_failed_or_interrupted_child_stops_remaining_runs(self):
        # Substitute only the external Python process, so the real batch script
        # handles activation, launch order and nonzero status propagation.
        for exit_code in (37, 130):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory(prefix="mine_batch_") as temp:
                root = Path(temp)
                (root / "scripts").mkdir()
                (root / ".venv/bin").mkdir(parents=True)
                script = root / "scripts/train_mine_ablation.sh"
                shutil.copyfile(SCRIPT, script)
                (root / "scripts/activate_train.sh").write_text(
                    "export VIRTUAL_ENV=" + shlex.quote(str(root / ".venv")) + "\n")
                invocation_log = root / "calls.jsonl"
                fake_python = root / ".venv/bin/python"
                fake_python.write_text(
                    "#!" + sys.executable + "\nimport json, pathlib, sys\n"
                    "path = pathlib.Path(" + repr(str(invocation_log)) + ")\n"
                    "with path.open('a') as stream: stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                    "sys.exit(" + str(exit_code) + " if len(path.read_text().splitlines()) == 2 else 0)\n")
                fake_python.chmod(0o755)
                result = subprocess.run(["bash", str(script), "--seeds", "10,20,30"],
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, exit_code, result.stderr)
                calls = [json.loads(line) for line in invocation_log.read_text().splitlines()]
                self.assertEqual(len(calls), 2)
                self.assertEqual([call[call.index("--run_name") + 1] for call in calls],
                                 ["full_seed10", "full_seed20"])
                self.assertIn("Remaining runs were not started", result.stderr)


if __name__ == "__main__":
    unittest.main()
