"""Exercise evaluation lifecycle and real control-loop hooks without a simulator."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import isaacgym  # Must precede torch.
import torch

from wheel_legged_gym.envs.nezha2.nezha2_env import Nezha2FreeEnv
from wheel_legged_gym.evaluation.env import EvaluationEnv
from wheel_legged_gym.evaluation.metrics import VectorEpisodeMetrics
from wheel_legged_gym.evaluation.terrain import EvaluationTerrain


def bare_env(n=4, quota=1):
    env = EvaluationEnv.__new__(EvaluationEnv)
    env.num_envs, env.num_dof, env.device = n, 2, "cpu"
    env.evaluation = NS(max_tilt_deg=60., command_vx=1., task="test_task", checkpoint="model_1.pt")
    env.cfg = NS(env=NS(fail_to_terminal_time_s=.6), terrain=NS(mesh_type="heightfield"))
    env.dt, env.max_episode_length = .02, 10
    env.sim_params = NS(dt=.005, gravity=NS(x=0., y=0., z=-9.81))
    env._evaluation_started, env._seed, env._episode_quota = True, 42, quota
    env.episode_counts = torch.zeros(n, dtype=torch.long)
    env.attempt_indices = torch.full((n,), -1, dtype=torch.long)
    env._episode_status, env._episode_reason = [None] * n, [""] * n
    env.completed_episodes, env.episode_sink = [], Mock()
    env.measurement_joint_names = ["joint_a", "joint_b"]
    env.control_rows, env.physics_rows = [], []
    env.metrics = VectorEpisodeMetrics(n, "cpu", trajectory_sink=env.control_rows.extend,
                                       physics_sink=env.physics_rows.extend, num_joints=2)
    env.root_states = torch.zeros(n, 13)
    env.root_states[:, 0], env.root_states[:, 1], env.root_states[:, 6] = 1., 5., 1.
    env.base_pos = env.root_states[:, :3]
    env.base_lin_vel, env.base_ang_vel = torch.zeros(n, 3), torch.zeros(n, 3)
    env.projected_gravity = torch.tensor([[0., 0., -1.]]).repeat(n, 1)
    env.contact_forces, env.termination_contact_indices = torch.zeros(n, 1, 3), torch.tensor([0])
    env.fail_buf = torch.zeros(n)
    env.episode_length_buf = torch.ones(n, dtype=torch.long)
    env.traversal_bounds = torch.tensor([[0., 10., 0., 10.]]).repeat(n, 1)
    env.goal_positions = torch.tensor([[9., 5., 0.]]).repeat(n, 1)
    env.goal_directions = torch.tensor([[1., 0.]]).repeat(n, 1)
    env.spawn_positions = env.root_states[:, :3].clone()
    env.commands = torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1)
    env._step_commands = env.commands[:, :3].clone()
    env.total_mass_kg = torch.full((n,), 15.)
    env.terrain_types = torch.zeros(n, dtype=torch.long)
    env.terrain = NS(type_names=("flat",))
    env.terrain_x_min, env.terrain_y_min = 0., 0.
    env.terrain_x_max, env.terrain_y_max = 10., 10.
    env._start(torch.arange(n))
    return env


def offline_energy(rows, num_joints):
    """Test-only reconstruction; the recorder must not perform this reduction."""
    return sum(row["dt_s"] * sum(row["torque_j%02d_nm" % j] *
               (row["joint_vel_before_j%02d_radps" % j] + row["joint_vel_after_j%02d_radps" % j]) * .5
               for j in range(num_joints)) for row in rows)


class EvaluationEnvTests(unittest.TestCase):
    def test_constructor_binds_physics_sink_and_validates_actual_joint_order(self):
        cfg = NS(env=NS(num_envs=1), evaluation=NS(),
                 terrain=NS(mesh_type="trimesh", curriculum=False),
                 commands=NS(curriculum=False, heading_command=False),
                 noise=NS(add_noise=False), init_state=NS(rand_init_dof=False),
                 domain_rand=NS(randomize_torque_scale=False))
        actual_names = ["hip", "wheel"]

        def build(self, *_args):
            self.num_envs, self.num_dof, self.num_actions, self.device = 1, 2, 2, "cpu"
            self.torques = torch.zeros(1, 2)
            self.dof_names = ["hip", "wheel"]
            self.envs, self.actor_handles = [object()], [0]
            self.gym = NS(get_actor_dof_names=Mock(return_value=actual_names))

        sink = Mock()
        with patch.object(Nezha2FreeEnv, "__init__", new=build):
            env = EvaluationEnv(cfg, None, None, "cpu", True, physics_sink=sink)
            self.assertEqual(env.measurement_joint_names, ["hip", "wheel"])
            self.assertEqual(env.metrics.num_joints, 2)
            self.assertIs(env.metrics.physics_sink, sink)
            actual_names.reverse()
            with self.assertRaisesRegex(ValueError, "ordering"):
                EvaluationEnv(cfg, None, None, "cpu", True, physics_sink=sink)

    def test_float32_physics_dt_does_not_add_a_spurious_timeout_step(self):
        env = EvaluationEnv.__new__(EvaluationEnv)
        env.sim_params = NS(dt=float(np.float32(.005)))
        env.cfg = NS(control=NS(decimation=4), normalization=NS(obs_scales=NS()),
                     rewards=NS(scales=NS()), commands=NS(ranges=NS()),
                     terrain=NS(mesh_type="trimesh"), env=NS(episode_length_s=2.),
                     domain_rand=NS(push_interval_s=1., rand_interval_s=1.,
                                    gravity_rand_interval_s=1., gravity_impulse_duration=.1))
        env._parse_cfg(env.cfg)
        self.assertEqual(env.max_episode_length, 100)
        env.cfg.env.episode_length_s = 2.001
        env._parse_cfg(env.cfg)
        self.assertEqual(env.max_episode_length, 101)

    def test_initial_reset_starts_at_time_zero_without_hidden_simulation(self):
        env = bare_env(1)
        env.metrics.active[:] = False
        env._evaluation_started = False
        env.cfg.normalization = NS(clip_observations=10.)
        env.obs_buf = torch.full((1, 130), 20.)
        env.privileged_obs_buf = torch.zeros(1, 1125)
        with patch.object(env, "reset_idx") as reset, \
                patch.object(env, "compute_observations") as observe, \
                patch.object(env, "step", side_effect=AssertionError("hidden physics step")):
            obs, critic = env.reset()
        reset.assert_called_once()
        observe.assert_called_once()
        self.assertEqual(tuple(obs.shape), (1, 130))
        self.assertEqual(tuple(critic.shape), (1, 1125))
        self.assertEqual(obs.max(), 10.)
        self.assertTrue(env.metrics.active[0])
        self.assertEqual(env.metrics.elapsed_s[0], 0.)
        self.assertEqual(env.metrics.physics_elapsed_s[0], 0.)
        with self.assertRaisesRegex(RuntimeError, "already started"):
            env.reset()

    def test_terminal_metrics_precede_reset_and_failure_precedes_goal_timeout(self):
        env = bare_env()
        env.root_states[:, 0] = torch.tensor([9., 9., 4., 9.])
        env.projected_gravity[1, 2] = 0.  # Goal + timeout + fall => failure.
        env.root_states[3, 1] = 11.  # Goal + timeout + outside => failure.
        env.episode_length_buf[:] = 10
        env.check_termination()
        self.assertEqual(env._episode_status, ["success", "failure", "timeout", "failure"])
        self.assertEqual(env._episode_reason, ["goal_crossed", "tilt_limit", "time_limit", "outside_terrain"])
        self.assertEqual(env.time_out_buf.tolist(), [False, False, True, False])
        self.assertEqual(env.metrics.control_steps.tolist(), [1] * 4)

        def physical_reset(ids):
            self.assertFalse(env.metrics.active[ids].any())
            self.assertEqual(len(env.completed_episodes), 4)
            env.root_states[ids, 0] = 999.

        with patch.object(Nezha2FreeEnv, "reset_idx", side_effect=physical_reset), \
                patch.object(env, "_sync_reset_state"):
            env.reset_idx(torch.arange(4))
        self.assertTrue(env.evaluation_complete)
        self.assertEqual([row["end_base_position"][0] - row["start_base_position"][0]
                          for row in env.completed_episodes], [8., 8., 3., 8.])
        self.assertNotIn("progress_m", env.completed_episodes[0])
        self.assertEqual(env.episode_counts.tolist(), [1] * 4)
        self.assertEqual(env.metrics.active.tolist(), [False] * 4)
        self.assertEqual(env.completed_episodes[0]["task"], "test_task")
        self.assertEqual(env.completed_episodes[0]["eval_seed"], 42)
        # Later resets of a finished slot must not add trials or call finish again.
        with patch.object(Nezha2FreeEnv, "reset_idx"), patch.object(env, "_sync_reset_state"):
            env.reset_idx(torch.tensor([0]))
        self.assertEqual(len(env.completed_episodes), 4)

    def test_repeated_episode_resets_do_not_leak_distance_or_velocity_jump(self):
        env = bare_env(1, quota=2)
        env.root_states[0, 0] = 9.
        env.root_states[0, 7] = 1.
        env.check_termination()

        def physical_reset(ids):
            env.root_states[ids, 0], env.root_states[ids, 7:13] = 1., 0.
            env.episode_length_buf[ids] = 0

        with patch.object(Nezha2FreeEnv, "reset_idx", side_effect=physical_reset), \
                patch.object(env, "_sync_reset_state"):
            env.reset_idx(torch.tensor([0]))
        self.assertFalse(env.evaluation_complete)
        self.assertTrue(env.metrics.active[0])
        self.assertEqual(env.metrics.control_steps[0], 0)
        self.assertEqual(env.metrics.physics_steps[0], 0)
        self.assertEqual(env.metrics.previous_velocity[0, 0], 0.)
        self.assertIsNone(env._episode_status[0])
        env.interrupt("test stop")
        self.assertEqual([r["status"] for r in env.completed_episodes], ["success", "interrupted"])
        self.assertEqual(env.completed_episodes[1]["start_base_position"],
                         env.completed_episodes[1]["end_base_position"])
        env.interrupt("again")
        self.assertEqual(len(env.completed_episodes), 2)

    def test_explicit_resets_allocate_unique_attempt_ids_without_using_quota(self):
        env = bare_env(1, quota=1)

        def physical_reset(ids):
            env.root_states[ids, 0], env.episode_length_buf[ids] = 1., 0

        with patch.object(Nezha2FreeEnv, "reset_idx", side_effect=physical_reset), \
                patch.object(env, "_sync_reset_state"):
            for _ in range(2):
                env.metrics.physics_step(torch.ones(1, 2), torch.zeros(1, 2), torch.ones(1, 2), .005)
                env.check_termination()
                env.reset_idx(torch.tensor([0]))
                self.assertEqual(env.episode_counts[0], 0)
                self.assertFalse(env.evaluation_complete)
            env.episode_length_buf[:] = env.max_episode_length
            env.check_termination()
            env.reset_idx(torch.tensor([0]))

        self.assertEqual([row["episode_index"] for row in env.completed_episodes], [0, 1, 2])
        self.assertEqual([row["status"] for row in env.completed_episodes],
                         ["interrupted", "interrupted", "timeout"])
        self.assertEqual([(row["episode_index"], row["control_step"]) for row in env.control_rows],
                         [(0, 1), (1, 1), (2, 1)])
        self.assertEqual([(row["episode_index"], row["physics_step"]) for row in env.physics_rows],
                         [(0, 1), (1, 1)])
        self.assertEqual(env.episode_counts[0], 1)
        self.assertTrue(env.evaluation_complete)

    def test_real_control_loop_records_every_physics_substep_actual_torque(self):
        env = bare_env(1)
        env.cfg.normalization = NS(clip_actions=100., clip_observations=100.)
        env.cfg.env.use_ref_actions = False
        env.cfg.control = NS(action_smoothness=False, decimation=3)
        env.cfg.domain_rand = NS(add_dof_lag=False, add_imu_lag=False, randomize_torque_delay=False,
                                action_delay=0., action_noise=0.)
        env.actions = torch.zeros(1, 2)
        env.dof_state = torch.zeros(2, 2)
        env.dof_state[:, 1] = torch.tensor([1., -2.])
        # Observation velocity deliberately differs; energy must use physical state.
        env.dof_vel = torch.full((1, 2), 999.)
        env.torques = torch.zeros(1, 2)
        env.envs_steps_buf = torch.zeros(1)
        env.render = env.pre_physics_step = env.leg_post_physics_step = Mock()
        env.compute_dof_vel = Mock()
        env._compute_torques = Mock(return_value=torch.tensor([[2., -3.]]))
        env.post_physics_step = Mock(return_value=(torch.tensor([], dtype=torch.long), None))
        env.obs_buf, env.privileged_obs_buf = torch.zeros(1, 130), None
        env.rew_buf, env.reset_buf, env.extras = torch.zeros(1), torch.zeros(1), {}
        env.sim = object()
        observed = []

        def simulate(_):
            observed.append((env.torques.clone(), env._omega_before.clone()))
            env.dof_state[:, 1] += 1.

        env.gym = NS(set_dof_actuation_force_tensor=Mock(), simulate=simulate,
                     fetch_results=Mock(), refresh_dof_state_tensor=Mock())
        with patch("wheel_legged_gym.envs.nezha2.legged_robot.gymtorch.unwrap_tensor", side_effect=lambda x: x):
            env.step(torch.tensor([[99., 99.]]))
        self.assertEqual(len(observed), 3)
        self.assertEqual([x[1].tolist() for x in observed], [[[1., -2.]], [[2., -1.]], [[3., 0.]]])
        self.assertEqual(len(env.physics_rows), 3)
        self.assertEqual([row["joint_vel_before_j00_radps"] for row in env.physics_rows], [1., 2., 3.])
        self.assertEqual([row["joint_vel_after_j00_radps"] for row in env.physics_rows], [2., 3., 4.])
        self.assertEqual([row["torque_j01_nm"] for row in env.physics_rows], [-3.] * 3)
        self.assertAlmostEqual(offline_energy(env.physics_rows, 2), .0975)
        self.assertFalse(hasattr(env.metrics, "energy_j"))
        self.assertAlmostEqual(env.metrics.physics_elapsed_s[0].item(), .015)
        self.assertEqual(env.gym.set_dof_actuation_force_tensor.call_count, 3)

    def test_episode_metadata_preserves_joint_order_gravity_and_state_frames(self):
        env = bare_env(1)
        env.root_states[0, :3] = torch.tensor([2., 3., 4.])
        env.root_states[0, 7:13] = torch.tensor([5., 6., 7., .1, .2, .3])
        env.base_lin_vel[0] = torch.tensor([8., 9., 10.])
        env.base_ang_vel[0] = torch.tensor([.4, .5, .6])
        env._finish(torch.tensor([0]), "interrupted", "test")
        row = env.completed_episodes[0]
        self.assertEqual(row["dof_names"], ["joint_a", "joint_b"])
        self.assertEqual(row["mass_kg"], 15.)
        self.assertEqual(row["gravity_vector_m_s2"], [0., 0., -9.81])
        self.assertEqual(row["goal_direction_xy"], [1., 0.])
        self.assertEqual(row["end_base_position"], [2., 3., 4.])
        self.assertEqual(row["end_velocity_world"], [5., 6., 7.])
        self.assertEqual(row["end_velocity_body"], [8., 9., 10.])
        self.assertEqual(row["end_orientation_xyzw"], [0., 0., 0., 1.])

    def test_spawn_height_heading_and_per_slot_rng_are_independent(self):
        cfg = NS(mesh_type="heightfield", type_names=("flat", "stairs_down"),
                 terrain_length=8., terrain_width=8., horizontal_scale=.1,
                 vertical_scale=.005, border_size=1.)

        def build():
            env = EvaluationEnv.__new__(EvaluationEnv)
            env.num_envs, env.device = 2, "cpu"
            env.evaluation = NS()
            env.terrain = EvaluationTerrain(cfg, 2)
            env._get_env_origins()
            env._spawn_rngs = [np.random.default_rng(42 + i) for i in range(2)]
            env.root_states = torch.zeros(2, 13)
            env.base_init_state = torch.tensor([0., 0., .55, 0., 0., 0., 1., 2., 3., 4., 5., 6., 7.])
            env.gym, env.sim = NS(set_actor_root_state_tensor_indexed=Mock()), object()
            return env

        first, second = build(), build()
        with patch("wheel_legged_gym.evaluation.env.gymtorch.unwrap_tensor", side_effect=lambda x: x):
            first._reset_root_states(torch.tensor([0, 1]))
            second._reset_root_states(torch.tensor([0, 1]))
            torch.testing.assert_close(first.root_states, second.root_states)
            first._reset_root_states(torch.tensor([0]))
            first._reset_root_states(torch.tensor([0]))
            first._reset_root_states(torch.tensor([1]))
            second._reset_root_states(torch.tensor([1]))
        torch.testing.assert_close(first.root_states[1], second.root_states[1])
        self.assertEqual(first.terrain_types.tolist(), [0, 1])
        self.assertTrue(torch.all(first.root_states[:, 7:13] == 0))
        torch.testing.assert_close(first.root_states[:, 3:7], torch.tensor([[0., 0., 0., 1.]]).repeat(2, 1))
        expected_ground = first.terrain.ground_heights(first.root_states[:, :2].numpy())
        np.testing.assert_allclose(first.root_states[:, 2].numpy(), expected_ground + .55, atol=1e-6)
        self.assertTrue(torch.all(first.goal_positions[:, 0] > first.root_states[:, 0]))

    def test_total_mass_includes_every_link(self):
        env = EvaluationEnv.__new__(EvaluationEnv)
        env.num_envs, env.device = 2, "cpu"
        props = [NS(mass=10.), NS(mass=2.), NS(mass=3.)]
        with patch.object(Nezha2FreeEnv, "_process_rigid_body_props", return_value=props):
            self.assertIs(env._process_rigid_body_props(props, 1), props)
        self.assertEqual(env.total_mass_kg.tolist(), [0., 15.])

    def test_reset_clears_physical_velocity_with_derived_velocity_observations(self):
        env = EvaluationEnv.__new__(EvaluationEnv)
        env.num_envs, env.num_dof = 2, 2
        env.cfg = NS(init_state=NS(rand_init_dof=False))
        env.dof_state = torch.zeros(4, 2)
        physical = env.dof_state.view(2, 2, 2)
        physical[:, :, 1] = torch.tensor([[10., 11.], [20., 21.]])
        env.dof_pos = physical[:, :, 0]
        env.dof_vel = torch.full((2, 2), 999.)  # Different storage, as with pos-diff.
        env.raw_default_dof_pos = torch.tensor([.1, .2])
        submitted = []
        env.sim = object()
        env.gym = NS(set_dof_state_tensor_indexed=lambda *args: submitted.append(physical.clone()))
        with patch("wheel_legged_gym.envs.nezha2.legged_robot.gymtorch.unwrap_tensor", side_effect=lambda x: x):
            env._reset_dofs(torch.tensor([1]))
        self.assertEqual(len(submitted), 1)
        torch.testing.assert_close(submitted[0][0, :, 1], torch.tensor([10., 11.]))
        torch.testing.assert_close(submitted[0][1, :, 1], torch.zeros(2))
        torch.testing.assert_close(submitted[0][1, :, 0], env.raw_default_dof_pos)
        torch.testing.assert_close(env.dof_vel[1], torch.zeros(2))

    def test_torque_randomization_defaults_preserved_and_explicitly_disabled(self):
        env = Nezha2FreeEnv.__new__(Nezha2FreeEnv)
        env.num_envs, env.num_dof, env.device = 1, 2, "cpu"
        domain = NS(add_lag=False, randomize_PD_factor=False,
                    randomize_coulomb_friction=False, randomize_motor_strength=False)
        env.cfg = NS(domain_rand=domain, control=NS(pos_action_scale=1., vel_action_scale=2., control_type="P"))
        env.default_dof_pos, env.dof_pos, env.dof_vel = (torch.zeros(1, 2) for _ in range(3))
        env.p_gains, env.d_gains = torch.ones(1, 2), torch.ones(1, 2)
        env.torque_limits, env.wheel_indices = torch.full((2,), 100.), [1]
        action = torch.ones(1, 2)
        with patch("wheel_legged_gym.envs.nezha2.nezha2_env.torch_rand_float", return_value=torch.full((1, 2), .9)) as rand:
            old = env._compute_torques(action)
            rand.assert_called_once_with(.8, 1.1, (1, 2), device="cpu")
            domain.randomize_torque_scale = False
            fixed = env._compute_torques(action)
            self.assertEqual(rand.call_count, 1)
        torch.testing.assert_close(old, torch.tensor([[.9, 1.8]]))
        torch.testing.assert_close(fixed, torch.tensor([[1., 2.]]))


if __name__ == "__main__":
    unittest.main()
