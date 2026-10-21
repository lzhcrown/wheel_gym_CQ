"""CPU regression tests; use the Isaac Gym training Python environment."""
import csv
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch
from collections import deque

import isaacgym  # Isaac Gym must be imported before torch.
import torch
from wheel_legged_gym import envs, algo
from wheel_legged_gym.utils import task_registry
from wheel_legged_gym.utils.helpers import class_to_dict, get_load_path
from wheel_legged_gym.experiments.chapter4_5_1 import CONFIGS
from wheel_legged_gym.experiments.chapter4_5_1.env import Chapter451Env
from wheel_legged_gym.metric_export import load_series, export_tensorboard
from wheel_legged_gym.training_metrics import TrainingSummaryWriter

torch.set_num_threads(1)


class FakeEnv:
    """Deterministic tiny rollout to exercise real PPO and logging on CPU."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_envs = 4
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions
        self.max_episode_length = 2
        self.episode_length_buf = torch.zeros(4, dtype=torch.long)
        self.obs = torch.randn(4, self.num_obs) * .1
        self.priv = torch.randn(4, self.num_privileged_obs) * .1

    def reset(self):
        return self.obs, self.priv

    def get_observations(self):
        return self.obs

    def get_privileged_observations(self):
        return self.priv

    def step(self, actions):
        self.obs = self.obs + .001
        self.priv = self.priv + .002
        return self.obs, self.priv, 1 - actions.square().mean(1), torch.ones(4), {
            'episode': {'rew_tracking_lin_vel': torch.tensor(.5)}}


class Chapter451Tests(unittest.TestCase):
    def test_shared_settings_and_isolation(self):
        base_env, base_train = task_registry.get_cfgs('exp451_baseline')
        common_env = class_to_dict(base_env)
        common_train = class_to_dict(base_train)
        env_differences = {'actor_true_state', 'num_est_v_h', 'num_est_prob',
                          'single_num_privileged_obs', 'num_privileged_obs',
                          'actor_input_stack', 'num_single_obs', 'num_observations'}
        for task in CONFIGS:
            cfg, train = task_registry.get_cfgs(task)
            e, t = class_to_dict(cfg), class_to_dict(train)
            for key in common_env:
                if key != 'env':
                    self.assertEqual(e[key], common_env[key], (task, key))
            self.assertEqual({k: v for k, v in e['env'].items() if k not in env_differences},
                             {k: v for k, v in common_env['env'].items() if k not in env_differences})
            for key, value in common_train['algorithm'].items():
                self.assertEqual(t['algorithm'][key], value, (task, key))
            for key, value in common_train['policy'].items():
                self.assertEqual(t['policy'][key], value, (task, key))
            for key, value in common_train['runner'].items():
                if key not in ('policy_class_name', 'algorithm_class_name', 'run_name'):
                    self.assertEqual(t['runner'][key], value, (task, key))
            self.assertEqual(t['seed'], common_train['seed'])
            self.assertIs(task_registry.get_task_class(task), Chapter451Env)
        base_env.rewards.scales.feet_air_time = 999
        base_train.policy.actor_hidden_dims[0] = 1
        again, train = task_registry.get_cfgs('exp451_baseline')
        self.assertEqual(again.rewards.scales.feet_air_time, 4.)
        self.assertEqual(train.policy.actor_hidden_dims, [512, 256, 128])

    def test_two_frame_observation_and_target_alignment(self):
        cfg, _ = task_registry.get_cfgs('exp451_actor_critic_truth')
        cfg.domain_rand.add_dof_lag = cfg.domain_rand.add_imu_lag = False
        cfg.domain_rand.randomize_obs_delay = False
        env = object.__new__(Chapter451Env)
        env.cfg, env.num_envs, env.device = cfg, 2, 'cpu'
        env.obs_scales = cfg.normalization.obs_scales
        env.commands_scale = torch.ones(3)
        for field, shape in {
            'base_height': (2,), 'root_states': (2, 13), 'measured_heights': (2, 187),
            'base_ang_vel': (2, 3), 'base_lin_vel': (2, 3), 'base_euler_xyz': (2, 3),
            'projected_gravity': (2, 3), 'commands': (2, 4), 'dof_pos': (2, 16),
            'default_dof_pos': (2, 16), 'dof_vel': (2, 16), 'actions': (2, 16),
            'dof_acc': (2, 16), 'torques': (2, 16), 'body_mass': (2,),
            'base_com': (2, 3), 'p_gains': (2, 16), 'd_gains': (2, 16),
            'contact_forces': (2, 17, 3),
        }.items():
            setattr(env, field, torch.zeros(shape))
        env.obs_history = deque([torch.zeros(2, 69) for _ in range(2)], maxlen=2)
        env.critic_history = deque([torch.zeros(2, 375) for _ in range(3)], maxlen=3)
        env.noise_scale_vec = env._get_noise_scale_vec(cfg)
        self.assertEqual(torch.count_nonzero(env.noise_scale_vec[-4:]).item(), 0)
        env.base_lin_vel[:] = torch.tensor([1., 2., 3.])
        env.base_height[:] = .5
        env.compute_observations()
        older = torch.tensor([2., 4., 6., 2.5]).expand(2, -1)
        env.base_lin_vel[:] = torch.tensor([4., 5., 6.])
        env.base_height[:] = .7
        env.compute_observations()
        newer = torch.tensor([8., 10., 12., 3.5]).expand(2, -1)
        self.assertEqual(env.obs_buf.shape, (2, 138))
        torch.testing.assert_close(env.obs_buf[:, 65:69], older)
        torch.testing.assert_close(env.obs_buf[:, 134:138], newer)
        torch.testing.assert_close(env.privileged_obs_buf.reshape(2, 3, 375)[:, -2:, -4:],
                                   torch.stack([older, newer], 1))
        from wheel_legged_gym.algo.state_estimate.actor_critic_est import ActorCritic_Estimator
        model = ActorCritic_Estimator(130, 130, 4, 1125, 16, estimated_frames=2,
                                      critic_frame_size=375, estimator_hidden_dims=[64, 64])
        targets = model.estimator.targets(env.privileged_obs_buf, 4)
        torch.testing.assert_close(targets, torch.cat([older, newer], 1))
        raw = torch.arange(260.).reshape(2, 130)
        packed = model.actor_observations(raw, targets)
        torch.testing.assert_close(packed[:, :65], raw[:, :65])
        torch.testing.assert_close(packed[:, 65:69], older)
        torch.testing.assert_close(packed[:, 69:134], raw[:, 65:])
        torch.testing.assert_close(packed[:, 134:], newer)
        output = targets.clone().requires_grad_()
        output.data[0, 0] += 1
        output.data[1, 4] += 2
        model.estimator.loss_fn(output, env.privileged_obs_buf, 4)['loss'].mean().backward()
        self.assertNotEqual(output.grad[0, 0].item(), 0)
        self.assertNotEqual(output.grad[1, 4].item(), 0)

    def test_six_runners_update_resume_and_export(self):
        for task in CONFIGS:
            with self.subTest(task=task), tempfile.TemporaryDirectory(prefix='exp451_test_') as tmp:
                cfg, train = task_registry.get_cfgs(task)
                train.runner.num_steps_per_env = 2
                train.runner.save_interval = 1
                train.algorithm.num_learning_epochs = 1
                train.algorithm.num_mini_batches = 1
                values = {**class_to_dict(train), **class_to_dict(cfg)}
                runner_class = getattr(algo, train.runner_class_name)
                runner = runner_class(FakeEnv(cfg), values, tmp, device='cpu')
                model = runner.alg.actor_critic
                expected = 130 if task in ('exp451_baseline', 'exp451_critic_truth') else 138
                self.assertEqual(model.actor[0].in_features, expected)
                self.assertEqual(model.critic[0].in_features, 1113 if task.endswith('baseline') else 1125)
                if hasattr(model, 'estimator'):
                    widths = [layer.out_features for layer in model.estimator.mlp_est.encoder if isinstance(layer, torch.nn.Linear)]
                    self.assertEqual(widths, train.policy.estimator_hidden_dims + [8])
                    self.assertEqual(model.estimator(torch.zeros(4, 130)).shape, (4, 8))
                    before = model.estimator.mlp_est.encoder[-1].weight.detach().clone()
                runner.learn(2)
                self.assertEqual(runner.current_learning_iteration, 2)
                self.assertTrue((Path(tmp) / 'model_1.pt').exists())
                self.assertTrue((Path(tmp) / 'model_2.pt').exists())
                if hasattr(model, 'estimator'):
                    after = model.estimator.mlp_est.encoder[-1].weight.detach()
                    self.assertTrue((before[:4] != after[:4]).any())
                    self.assertTrue((before[4:] != after[4:]).any())
                runner.writer.close()
                restored = runner_class(FakeEnv(cfg), values, tmp, device='cpu')
                restored.load(str(Path(tmp) / 'model_2.pt'))
                self.assertEqual(restored.current_learning_iteration, 2)
                self.assertEqual(restored.alg.learning_rate, runner.alg.learning_rate)
                self.assertEqual(restored.alg.optimizer.state_dict()['param_groups'],
                                 runner.alg.optimizer.state_dict()['param_groups'])
                if hasattr(model, 'estimator'):
                    self.assertEqual(len(restored.alg.estimator_optimizer.state), len(runner.alg.estimator_optimizer.state))
                    self.assertGreater(len(restored.alg.estimator_optimizer.state), 0)
                    input_obs = torch.randn(4, 130)
                    torch.testing.assert_close(restored.alg.actor_critic.act_inference(input_obs, input_obs),
                                               model.act_inference(input_obs, input_obs))
                restored.learn(1)
                restored.writer.close()
                steps, streamed_values = load_series(tmp, 'Train/mean_reward')
                self.assertEqual(steps, [0., 1., 2.])
                manifest = export_tensorboard(tmp)
                self.assertIn('Train/mean_reward', manifest)
                exported_steps, exported_values = load_series(tmp, 'Train/mean_reward')
                self.assertEqual(exported_steps, steps)
                for streamed, exported in zip(streamed_values, exported_values):
                    self.assertAlmostEqual(streamed, exported, places=5)
                self.assertFalse((Path(tmp) / 'plots').exists())
                self.assertFalse(list(Path(tmp).rglob('*.png')))
                self.assertFalse(list(Path(tmp).rglob('*.pdf')))

    def test_raw_csv_and_multi_run_export(self):
        with tempfile.TemporaryDirectory(prefix='exp451_metric_test_') as tmp:
            runs = [Path(tmp) / name for name in ('a', 'b')]
            expected = {'Train/mean_reward': [2., 4., 12.],
                        'Train/mean reward': [3., 9., 1.]}
            for run in runs:
                writer = TrainingSummaryWriter(str(run))
                for tag, values in expected.items():
                    for step, value in enumerate(values):
                        writer.add_scalar(tag, value, step, walltime=1000. + step)
                writer.close()
                for tag, values in expected.items():
                    self.assertEqual(load_series(run, tag), ([0., 1., 2.], values))
                # Remove the streamed CSVs: the CLI must recover them from events.
                for path in (run / 'csv').iterdir():
                    path.unlink()
            script = Path(__file__).resolve().parents[1] / 'wheel_legged_gym/scripts/tensorboard_data_ex.py'
            subprocess.run([sys.executable, str(script), *map(str, runs)],
                           check=True, capture_output=True, text=True)
            for run in runs:
                manifest = json.loads((run / 'csv/tags.json').read_text())
                self.assertEqual(set(manifest), set(expected))
                self.assertEqual(len(set(manifest.values())), 2)
                for tag, values in expected.items():
                    self.assertEqual(load_series(run, tag), ([0., 1., 2.], values))
                    with (run / 'csv' / manifest[tag]).open(newline='') as stream:
                        reader = csv.DictReader(stream)
                        self.assertEqual(reader.fieldnames, ['wall_time', 'step', 'value'])
                        rows = list(reader)
                    self.assertEqual([float(row['wall_time']) for row in rows], [1000., 1001., 1002.])
                    self.assertEqual([float(row['value']) for row in rows], values)
                self.assertFalse((run / 'plots').exists())
            self.assertFalse(list(Path(tmp).rglob('*.png')))
            self.assertFalse(list(Path(tmp).rglob('*.pdf')))

    def test_cli_overrides_and_actual_config_snapshot(self):
        from wheel_legged_gym.utils.helpers import get_args
        from wheel_legged_gym.utils.task_registry import TaskRegistry
        with patch('sys.argv', ['train.py', '--task', 'exp451_est_64_64', '--num_envs', '4',
                               '--max_iterations', '7', '--save_interval', '3', '--seed', '123',
                               '--rl_device', 'cpu', '--sim_device', 'cpu']):
            args = get_args(defaults={'task': 'exp451_baseline', 'headless': True})
        self.assertEqual(args.task, 'exp451_est_64_64')
        self.assertTrue(args.headless)
        registry = TaskRegistry()
        cfg, train = task_registry.get_cfgs(args.task)
        registry.register(args.task, lambda **kw: FakeEnv(kw['cfg']), cfg, train)
        with tempfile.TemporaryDirectory(prefix='exp451_snapshot_test_') as tmp:
            env, _ = registry.make_env(args.task, args=args)
            runner, _ = registry.make_alg_runner(env, name=args.task, args=args, log_root=tmp)
            registry.save_cfgs(args.task)
            run = Path(runner.log_dir)
            saved_env = json.loads((run / 'env_cfg.json').read_text())
            saved_train = json.loads((run / 'train_cfg.json').read_text())
            cli = json.loads((run / 'cli_args.json').read_text())
            info = json.loads((run / 'run_info.json').read_text())
            self.assertEqual(saved_env['seed'], 123)
            self.assertEqual(saved_env['env']['num_envs'], 4)
            self.assertEqual(saved_train['runner']['max_iterations'], 7)
            self.assertEqual(saved_train['runner']['save_interval'], 3)
            self.assertEqual(cli['task'], args.task)
            self.assertEqual(info['actor_input_dim'], 138)
            self.assertEqual(info['critic_input_dim'], 1125)
            self.assertTrue((run / 'source/wheel_legged_gym/experiments/chapter4_5_1/common.py').exists())

    def test_checkpoint_selection_ignores_model_metadata_and_empty_runs(self):
        with tempfile.TemporaryDirectory(prefix='exp451_checkpoint_test_') as tmp:
            root = Path(tmp)
            run = root / '20261021_baseline'
            run.mkdir()
            for name in ('model_9.pt', 'model_100.pt', 'model_source.json'):
                (run / name).touch()
            (run / 'run_info.json').write_text(json.dumps({'task': 'exp451_baseline'}))
            other = root / '20261022_est_64_64'
            other.mkdir()
            (other / 'model_1.pt').touch()
            (other / 'run_info.json').write_text(json.dumps({'task': 'exp451_est_64_64'}))
            self.assertEqual(get_load_path(tmp, task='exp451_baseline'), str(run / 'model_100.pt'))
            with self.assertRaises(ValueError):
                get_load_path(tmp, other.name, task='exp451_baseline')
            (other / 'model_1.pt').unlink()
            (root / 'zz_comparison').mkdir()
            self.assertEqual(get_load_path(tmp), str(run / 'model_100.pt'))
            self.assertEqual(get_load_path(tmp, run.name, 9), str(run / 'model_9.pt'))


if __name__ == '__main__':
    unittest.main()
