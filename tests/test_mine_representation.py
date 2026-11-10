"""Numerical checks for MINE's history gating and auxiliary representation path."""
import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import isaacgym  # Import the simulator bindings before torch.
import torch
import torch.nn.functional as F

from wheel_legged_gym.algo.Mine.actor_critic_mine import MINEActorCritic
from wheel_legged_gym.algo.Mine.ppo_mine import MINEPPO
from wheel_legged_gym.algo.Mine.on_policy_runner_mine import MINEOnPolicyRunner
from wheel_legged_gym import envs
from wheel_legged_gym.utils import task_registry
from wheel_legged_gym.utils.helpers import class_to_dict


torch.set_num_threads(1)


def make_model():
    cfg, train = task_registry.get_cfgs('nezha2_mine')
    with contextlib.redirect_stdout(io.StringIO()):
        model = MINEActorCritic(
            cfg.env.num_observations, cfg.env.num_privileged_obs,
            cfg.env.num_single_obs, cfg.env.single_num_privileged_obs,
            cfg.env.num_est_prob, cfg.env.num_actions,
            **class_to_dict(train.policy))
    return model


class MineRepresentationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_runner_keeps_stacked_critic_but_passes_single_frame_target_size(self):
        cfg, train = task_registry.get_cfgs('nezha2_mine')
        environment = SimpleNamespace(
            cfg=cfg, num_envs=4, num_obs=cfg.env.num_observations,
            num_privileged_obs=cfg.env.num_privileged_obs,
            num_actions=cfg.env.num_actions, reset=lambda: (None, None))
        with contextlib.redirect_stdout(io.StringIO()):
            runner = MINEOnPolicyRunner(environment, class_to_dict(train), device='cpu')
        self.assertEqual(runner.alg.actor_critic.critic[0].in_features, 1125)
        self.assertEqual(runner.alg.storage.privileged_observations.shape[-1], 1125)
        self.assertEqual(runner.alg.storage.next_privileged_observations.shape[-1], 1125)
        self.assertEqual(runner.alg.actor_critic.estimator.num_one_step_privileged_obs, 375)
        self.assertEqual(runner.alg.actor_critic.estimator.target[0].in_features, 375)

    def test_gate_receives_both_frames_and_responds_to_each(self):
        model = make_model()
        estimator = model.estimator
        self.assertEqual(estimator.mode_gate[0].in_features, 130)
        history = torch.linspace(-.4, .6, 4 * 130).reshape(4, 130)
        captured = []
        handle = estimator.mode_gate.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0].detach().clone()))
        try:
            baseline = estimator._encode_history(history)[2]
            changed_old = history.clone()
            changed_old[:, :65] += .3
            old_probs = estimator._encode_history(changed_old)[2]
            changed_latest = history.clone()
            changed_latest[:, 65:] -= .3
            latest_probs = estimator._encode_history(changed_latest)[2]
        finally:
            handle.remove()
        torch.testing.assert_close(captured[0], history)
        torch.testing.assert_close(captured[1], changed_old)
        torch.testing.assert_close(captured[2], changed_latest)
        self.assertGreater((baseline - old_probs).abs().max().item(), 1e-6)
        self.assertGreater((baseline - latest_probs).abs().max().item(), 1e-6)
        torch.testing.assert_close(baseline.sum(-1), torch.ones(4))

    def test_actor_uses_latest_frame_in_sampling_and_inference(self):
        model = make_model()
        history = torch.randn(4, 130)
        history[:, :65] -= 5
        history[:, 65:] += 5
        captured = []
        handle = model.actor.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0].detach().clone()))
        try:
            model.act(history)
            sampled_mean = model.action_mean.detach().clone()
            predicted_mean = model.act_inference(history)
        finally:
            handle.remove()
        self.assertEqual(len(captured), 2)
        vel, latent = model.estimator(history)
        for actor_input in captured:
            self.assertEqual(actor_input.shape, (4, 85))
            torch.testing.assert_close(actor_input[:, :65], history[:, 65:])
            torch.testing.assert_close(actor_input[:, 65:69], vel)
            torch.testing.assert_close(actor_input[:, 69:], latent)
        torch.testing.assert_close(predicted_mean, sampled_mean)

    def test_normalization_matches_auxiliary_training_and_prototype_scaling(self):
        estimator = make_model().estimator
        history = torch.randn(8, 130)
        critic_history = torch.randn(8, 1125)
        next_critic_history = torch.randn(8, 1125)
        original_prototypes = estimator.mode_prototypes.detach().clone()
        vel, latent, probabilities, normalized = estimator._encode_history(history)
        torch.testing.assert_close(normalized.norm(dim=-1), torch.ones(3))
        torch.testing.assert_close(latent.norm(dim=-1), torch.ones(8))
        encoded = estimator.encode(history)
        detached = estimator(history)
        for actual in (encoded, detached):
            torch.testing.assert_close(actual[0], vel)
            torch.testing.assert_close(actual[1], latent)
        self.assertFalse(detached[0].requires_grad)
        self.assertFalse(detached[1].requires_grad)
        self.assertTrue(encoded[1].requires_grad)
        torch.testing.assert_close(estimator.mode_prototypes, original_prototypes)

        # Observe the actual student scores used by the swap objective, rather
        # than only checking that update() calls a common helper.
        score_inputs = []
        mode_inputs = []
        real_log_softmax = F.log_softmax
        real_cross_entropy = F.cross_entropy

        def capture_scores(values, *args, **kwargs):
            score_inputs.append(values.detach().clone())
            return real_log_softmax(values, *args, **kwargs)

        def capture_modes(values, *args, **kwargs):
            mode_inputs.append(values.detach().clone())
            return real_cross_entropy(values, *args, **kwargs)

        expected_scores = latent.detach() @ F.normalize(estimator.proto.weight.detach(), dim=-1).T
        expected_modes = latent.detach() @ normalized.detach().T
        with patch('wheel_legged_gym.algo.Mine.estimator_mine.F.log_softmax', side_effect=capture_scores), \
             patch('wheel_legged_gym.algo.Mine.estimator_mine.F.cross_entropy', side_effect=capture_modes):
            estimator.update(history, critic_history, next_critic_history, lr=0.)
        torch.testing.assert_close(score_inputs[0], expected_scores / estimator.temperature)
        torch.testing.assert_close(mode_inputs[0], expected_modes)
        torch.testing.assert_close(estimator.mode_prototypes, original_prototypes)

        # Rescaling only one prototype must not change its contribution relative
        # to the other modes after per-prototype normalization.
        with torch.no_grad():
            estimator.mode_prototypes[1].mul_(7.)
        scaled_encoded = estimator.encode(history)
        scaled_forward = estimator(history)
        for actual in (scaled_encoded, scaled_forward):
            torch.testing.assert_close(actual[0], vel)
            torch.testing.assert_close(actual[1], latent, atol=1e-6, rtol=1e-5)
        score_inputs.clear()
        mode_inputs.clear()
        with patch('wheel_legged_gym.algo.Mine.estimator_mine.F.log_softmax', side_effect=capture_scores), \
             patch('wheel_legged_gym.algo.Mine.estimator_mine.F.cross_entropy', side_effect=capture_modes):
            estimator.update(history, critic_history, next_critic_history, lr=0.)
        torch.testing.assert_close(score_inputs[0], expected_scores / estimator.temperature,
                                   atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(mode_inputs[0], expected_modes, atol=1e-6, rtol=1e-5)

    def test_ppo_detaches_representation_but_auxiliary_objective_trains_it(self):
        model = make_model()
        history = torch.randn(8, 130)
        critic_history = torch.randn(8, 1125)
        model.zero_grad(set_to_none=True)
        actions = model.act(history).detach()
        advantage = torch.linspace(-1., 1., 8)
        objective = -(model.get_actions_log_prob(actions) * advantage).mean()
        objective = objective + model.evaluate(critic_history).square().mean() - .01 * model.entropy.mean()
        objective.backward()
        self.assertGreater(model.actor[0].weight.grad.abs().sum().item(), 0.)
        self.assertGreater(model.critic[0].weight.grad.abs().sum().item(), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in model.estimator.parameters()))

        model.zero_grad(set_to_none=True)
        model.estimator.update(history, critic_history, torch.randn(8, 1125), lr=0.)
        for name, parameter in {
            'source': model.estimator.encoder[0].weight,
            'source_latent_head': model.estimator.encoder[-1].weight,
            'target': model.estimator.target[0].weight,
            'gate': model.estimator.mode_gate[0].weight,
            'mode_prototypes': model.estimator.mode_prototypes,
            'swap_prototypes': model.estimator.proto.weight,
        }.items():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            gradient = parameter.grad[4:] if name == 'source_latent_head' else parameter.grad
            self.assertGreater(gradient.abs().sum().item(), 0., name)

    def test_real_ppo_target_is_current_full_frame_even_for_terminal_transitions(self):
        model = make_model()
        self.assertTrue(model.estimator.is_privileged_obs)
        self.assertEqual(model.estimator.num_one_step_privileged_obs, 375)
        self.assertEqual(model.estimator.target[0].in_features, 375)
        algorithm = MINEPPO(model, num_learning_epochs=1, num_mini_batches=1,
                            learning_rate=0., desired_kl=None, device='cpu')
        algorithm.init_storage(4, 2, [130], [1125], [16])
        expected_frames, expected_states = [], []
        per_coordinate = torch.arange(375, dtype=torch.float)[None, :] / 1000.
        per_environment = torch.arange(4, dtype=torch.float)[:, None] / 10.

        def frame(value):
            return value + per_coordinate + per_environment

        for step in range(2):
            # Current=[10,20,30], next=[20,30,40]. Each frame also has
            # distinct coordinates beyond the proprioceptive first 65 values.
            current = torch.cat([frame(value + step) for value in (10., 20., 30.)], 1)
            following = torch.cat([frame(value + step) for value in (20., 30., 40.)], 1)
            # Emulate the runner's terminal replacement: environment 2 must
            # carry the old episode's final state, not its reset observation.
            following[2, -375:] = frame(70. + step)[2]
            history = torch.randn(4, 130)
            expected_frames.append(current[:, -375:].clone())
            expected_states.append(following[:, -4:].clone())
            with torch.inference_mode():
                algorithm.act(history, current)
                algorithm.process_env_step(torch.tensor([1., .2, -.1, .8]),
                                           torch.tensor([False, False, True, False]),
                                           {}, following)
        with torch.inference_mode():
            algorithm.compute_returns(following)

        target_inputs, state_targets = [], []
        target_hook = model.estimator.target.register_forward_pre_hook(
            lambda module, inputs: target_inputs.append(inputs[0].detach().clone()))
        original_mse = F.mse_loss

        def capture_state_target(predicted, target, *args, **kwargs):
            state_targets.append(target.detach().clone())
            return original_mse(predicted, target, *args, **kwargs)

        try:
            with patch('wheel_legged_gym.algo.Mine.estimator_mine.F.mse_loss',
                       side_effect=capture_state_target):
                algorithm.update()
        finally:
            target_hook.remove()

        observed = torch.cat(target_inputs)
        expected = torch.cat(expected_frames)
        self.assertEqual(observed.shape, (8, 375))
        torch.testing.assert_close(observed[observed[:, 0].argsort()],
                                   expected[expected[:, 0].argsort()])
        # Target uses current p_t, while explicit regression intentionally
        # retains the existing next-state velocity/height supervision.
        observed_states = torch.cat(state_targets)
        expected_states = torch.cat(expected_states)
        self.assertEqual(observed_states.shape, (8, 4))
        torch.testing.assert_close(observed_states[observed_states[:, 0].argsort()],
                                   expected_states[expected_states[:, 0].argsort()])


if __name__ == '__main__':
    unittest.main()
