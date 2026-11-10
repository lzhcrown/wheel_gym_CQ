"""Exercise the retained MINE learner on CPU, without starting a simulator."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import isaacgym  # Isaac Gym must be imported before torch.
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from wheel_legged_gym import algo, envs
from wheel_legged_gym.metric_export import export_tensorboard, load_series
from wheel_legged_gym.utils import task_registry
from wheel_legged_gym.utils.helpers import class_to_dict


torch.set_num_threads(1)


class TinyMineEnv:
    """Return the seven-value MINE API, including pre-reset terminal states."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_envs = 4
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions
        self.max_episode_length = 2
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.obs = torch.randn(self.num_envs, self.num_obs) * .1
        self.critic_obs = torch.randn(self.num_envs, self.num_privileged_obs) * .1
        self.terminal_ids = torch.tensor([0, 2])
        self.terminal_obs = None

    def reset(self):
        return self.obs, self.critic_obs

    def get_observations(self):
        return self.obs

    def get_privileged_observations(self):
        return self.critic_obs

    def step(self, actions, MINE=False):
        if not MINE:
            raise AssertionError('MINE runner must request terminal observations')
        if not torch.isfinite(actions).all():
            raise AssertionError('Policy actions must remain finite')
        self.obs = self.obs + .001
        self.critic_obs = self.critic_obs + .002
        self.terminal_obs = self.critic_obs[self.terminal_ids].clone() + .3
        reward = torch.ones(self.num_envs)
        done = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
        info = {'episode': {'rew_tracking_lin_vel': torch.tensor(.5)},
                'time_outs': torch.tensor([1., 0., 0., 0.])}
        return (self.obs, self.critic_obs, reward, done, info,
                self.terminal_ids, self.terminal_obs)


def runner_settings():
    cfg, train = task_registry.get_cfgs('nezha2_mine')
    train.runner.num_steps_per_env = 2
    train.runner.save_interval = 1
    train.algorithm.num_learning_epochs = 1
    train.algorithm.num_mini_batches = 1
    values = {**class_to_dict(train), **class_to_dict(cfg)}
    return cfg, values


class MineTrainingTests(unittest.TestCase):
    def test_real_gated_learner_terminal_states_resume_and_csv(self):
        torch.manual_seed(10)
        cfg, values = runner_settings()
        self.assertEqual((cfg.env.num_observations, cfg.env.num_privileged_obs), (130, 1125))
        self.assertEqual(values['runner']['policy_class_name'], 'MINEActorCritic')
        self.assertEqual(values['runner']['algorithm_class_name'], 'MINEPPO')
        with tempfile.TemporaryDirectory(prefix='mine_training_') as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            (root / 'train_cfg.json').write_text(json.dumps(values))
            env = TinyMineEnv(cfg)
            runner = algo.MINEOnPolicyRunner(env, values, temp, device='cpu')
            model = runner.alg.actor_critic
            self.assertEqual(model.estimator.mode_gate[-1].out_features, 3)
            self.assertEqual(model.estimator.mode_prototypes.shape, (3, 16))
            self.assertEqual(model.actor[0].in_features, 85)
            self.assertEqual(model.critic[0].in_features, 1125)
            parameters = {
                'actor': model.actor[0].weight,
                'critic': model.critic[0].weight,
                'source': model.estimator.encoder[0].weight,
                'target': model.estimator.target[0].weight,
                'gate': model.estimator.mode_gate[0].weight,
                'modes': model.estimator.mode_prototypes,
            }
            initial = {name: parameter.detach().clone() for name, parameter in parameters.items()}
            original_process = runner.alg.process_env_step
            terminal_checks = []

            def inspect_terminal(rewards, dones, infos, next_critic):
                torch.testing.assert_close(next_critic[env.terminal_ids], env.terminal_obs)
                torch.testing.assert_close(next_critic[[1, 3]], env.critic_obs[[1, 3]])
                terminal_checks.append(True)
                return original_process(rewards, dones, infos, next_critic)

            with patch.object(runner.alg, 'process_env_step', side_effect=inspect_terminal):
                runner.learn(2)
            self.assertEqual(len(terminal_checks), 4)
            self.assertEqual(runner.current_learning_iteration, 2)
            for name, parameter in parameters.items():
                self.assertTrue(torch.isfinite(parameter).all(), name)
                self.assertTrue(torch.any(parameter.detach() != initial[name]), name)
            for iteration in (1, 2):
                checkpoint = torch.load(root / ('model_%d.pt' % iteration), map_location='cpu')
                self.assertEqual(checkpoint['iter'], iteration)
            runner.writer.close()

            restored = algo.MINEOnPolicyRunner(TinyMineEnv(cfg), values, temp, device='cpu')
            # A CPU-only machine must load checkpoints without an unbound device.
            with patch('torch.cuda.is_available', return_value=False):
                restored.load(root / 'model_2.pt')
            self.assertEqual(restored.current_learning_iteration, 2)
            self.assertEqual(restored.tot_timesteps, runner.tot_timesteps)
            self.assertEqual(restored.tot_time, runner.tot_time)
            self.assertEqual(restored.alg.learning_rate, runner.alg.learning_rate)
            self.assertEqual(restored.alg.actor_critic.estimator.learning_rate,
                             model.estimator.learning_rate)
            for old_optimizer, new_optimizer in (
                (runner.alg.optimizer, restored.alg.optimizer),
                (model.estimator.optimizer, restored.alg.actor_critic.estimator.optimizer),
            ):
                old, new = old_optimizer.state_dict(), new_optimizer.state_dict()
                self.assertTrue(old['state'])
                self.assertEqual(old['param_groups'], new['param_groups'])
                self.assertEqual(set(old['state']), set(new['state']))
                for key in old['state']:
                    for state_name in old['state'][key]:
                        torch.testing.assert_close(old['state'][key][state_name], new['state'][key][state_name])
            obs = torch.randn(4, 130)
            torch.testing.assert_close(model.act_inference(obs),
                                       restored.alg.actor_critic.act_inference(obs))
            restored.learn(1)
            restored.writer.close()
            self.assertEqual(restored.current_learning_iteration, 3)

            manifest = json.loads((root / 'csv/tags.json').read_text())
            self.assertEqual(manifest, {'Train/mean_reward': 'Train_mean_reward.csv'})
            self.assertEqual({p.name for p in (root / 'csv').glob('*.csv')},
                             {'Train_mean_reward.csv'})
            steps, rewards = load_series(root, 'Train/mean_reward')
            self.assertEqual(steps, [0., 1., 2.])
            self.assertEqual(rewards, [1., 1., 1.])
            with (root / 'csv/Train_mean_reward.csv').open() as stream:
                rows = list(csv.DictReader(stream))
                self.assertEqual(set(rows[0]), {'wall_time', 'step', 'value'})
            events = EventAccumulator(str(root)).Reload()
            self.assertIn('Loss/Swap Loss', events.Tags()['scalars'])
            self.assertIn('Train/mean_reward/time', events.Tags()['scalars'])
            self.assertEqual(export_tensorboard(root), manifest)
            self.assertEqual(load_series(root, 'Train/mean_reward'), (steps, rewards))
            self.assertEqual({p.name for p in (root / 'csv').glob('*.csv')},
                             {'Train_mean_reward.csv'})

    def test_legacy_checkpoint_and_training_without_log_directory(self):
        cfg, values = runner_settings()
        with tempfile.TemporaryDirectory(prefix='mine_legacy_') as temp, contextlib.redirect_stdout(io.StringIO()):
            runner = algo.MINEOnPolicyRunner(TinyMineEnv(cfg), values, log_dir=None, device='cpu')
            runner.learn(1)
            self.assertEqual(runner.current_learning_iteration, 1)
            self.assertIsNone(runner.writer)
            path = Path(temp) / 'legacy.pt'
            runner.save(path)
            checkpoint = torch.load(path, map_location='cpu')
            for key in ('learning_rate', 'estimator_learning_rate', 'tot_timesteps', 'tot_time'):
                checkpoint.pop(key)
            torch.save(checkpoint, path)
            restored = algo.MINEOnPolicyRunner(TinyMineEnv(cfg), values, log_dir=None, device='cpu')
            restored.load(path)
            self.assertEqual(restored.alg.learning_rate, runner.alg.optimizer.param_groups[0]['lr'])
            self.assertEqual(restored.alg.actor_critic.estimator.learning_rate,
                             runner.alg.actor_critic.estimator.optimizer.param_groups[0]['lr'])
            self.assertEqual(restored.current_learning_iteration, 1)


if __name__ == '__main__':
    unittest.main()
