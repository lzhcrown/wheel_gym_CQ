"""CPU checks for the gated policy export and replay checkpoint selection."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import isaacgym  # Isaac Gym must be imported before torch.
import torch

from wheel_legged_gym.algo.Mine.actor_critic_mine import MINEActorCritic
from wheel_legged_gym.scripts.play import MineInferencePolicy, get_play_args, resolve_checkpoint
from wheel_legged_gym.utils.helpers import update_cfg_from_args


class MineExportTests(unittest.TestCase):
    def test_export_preserves_ablation_masks_and_gate_bypass(self):
        variants = {
            "full": {},
            "no_latent_input": {"use_latent_input": False},
            "no_state_input": {"use_state_input": False},
            "no_gate": {"use_mode_gate": False, "mode_loss_coef": 0.0},
            "no_mode_loss": {"mode_loss_coef": 0.0},
        }
        for name, flags in variants.items():
            with self.subTest(variant=name), contextlib.redirect_stdout(io.StringIO()):
                torch.manual_seed(43)
                model = MINEActorCritic(
                    130, 1125, 65, 375, 4, 16,
                    actor_hidden_dims=[32, 16], critic_hidden_dims=[32, 16],
                    enc_hidden_dims=[32, 16], tar_hidden_dims=[32],
                    is_privileged_obs=True, **flags).eval()
                observations = torch.randn(5, 130)
                with tempfile.TemporaryDirectory() as directory:
                    path = MineInferencePolicy(model).export(directory)
                    exported = torch.jit.load(str(path))
                    with torch.inference_mode():
                        for batch in (observations[:1], observations):
                            torch.testing.assert_close(exported(batch), model.act_inference(batch))

    def test_export_matches_gated_inference_for_multiple_batches(self):
        torch.manual_seed(23)
        with contextlib.redirect_stdout(io.StringIO()):
            actor_critic = MINEActorCritic(
                130, 1125, 65, 375, 4, 16,
                actor_hidden_dims=[32, 16], critic_hidden_dims=[32, 16],
                enc_hidden_dims=[32, 16], tar_hidden_dims=[32],
                is_privileged_obs=True,
            ).eval()
        # Give gate outputs distinct, non-unit prototypes so omitting the gate
        # cannot accidentally produce the same normalized latent.
        with torch.no_grad():
            actor_critic.estimator.mode_prototypes.copy_(
                torch.arange(48, dtype=torch.float32).reshape(3, 16) / 13 - 1
            )
        observations = torch.randn(7, 130)
        with torch.inference_mode():
            expected = actor_critic.act_inference(observations)
            parts = actor_critic.estimator.encoder(observations)
            ungated = actor_critic.actor(torch.cat((
                observations[:, -65:], parts[:, :4],
                torch.nn.functional.normalize(parts[:, 4:], dim=-1),
            ), dim=-1))
        self.assertFalse(torch.allclose(expected, ungated, atol=1e-5))
        original = {key: value.clone() for key, value in actor_critic.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            exported = MineInferencePolicy(actor_critic).export(directory)
            loaded = torch.jit.load(str(exported))
            with torch.inference_mode():
                for batch in (observations[:1], observations[:3], observations):
                    torch.testing.assert_close(loaded(batch), actor_critic.act_inference(batch))
            self.assertEqual([path.name for path in Path(directory).iterdir()], ["policy_mine.pt"])
        for key, value in actor_critic.state_dict().items():
            torch.testing.assert_close(value, original[key])

    def test_replay_honors_explicit_run_checkpoint_and_log_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, step in (("requested", 12), ("zz_newer", 999)):
                run = root / name
                run.mkdir()
                (run / "run_info.json").write_text(json.dumps({"task": "nezha2_mine"}))
                (run / "model_{}.pt".format(step)).touch()
            args = get_play_args([
                "--log-root", directory, "--experiment_name", "custom",
                "--load_run", "requested", "--checkpoint", "12",
                "--headless", "--num_envs", "2", "--play-steps", "7",
                "--rl_device", "cpu", "--sim_device", "cpu",
            ])
            config = SimpleNamespace(seed=10, runner=SimpleNamespace(
                experiment_name="default", load_run=-1, checkpoint=-1,
            ))
            _, config = update_cfg_from_args(None, config, args)
            self.assertEqual(args.task, "nezha2_mine")
            self.assertEqual(args.num_envs, 2)
            self.assertEqual(args.play_steps, 7)
            self.assertEqual(config.runner.experiment_name, "custom")
            self.assertEqual(resolve_checkpoint(args, config), root / "requested/model_12.pt")
            self.assertEqual(sorted(path.name for path in root.iterdir()), ["requested", "zz_newer"])


if __name__ == "__main__":
    unittest.main()
