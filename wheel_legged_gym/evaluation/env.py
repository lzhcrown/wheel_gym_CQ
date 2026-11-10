"""Independent, fixed-terrain evaluation with measurements before automatic reset."""
import math
from types import SimpleNamespace

import numpy as np
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse
import torch

from wheel_legged_gym.envs.nezha2.nezha2_env import Nezha2FreeEnv
from wheel_legged_gym.envs.nezha2.legged_robot import get_euler_xyz_tensor
from .metrics import VectorEpisodeMetrics
from .terrain import EvaluationTerrain


class EvaluationEnv(Nezha2FreeEnv):
    """Run a fixed number of independent traversals for each simulation slot.

    Construction does not start a measured episode. Call ``reset()`` once, then
    step until ``evaluation_complete``. Each slot keeps its terrain type and its
    own spawn RNG, including when other robots finish earlier. The inherited
    actor factory uses collision group ``env_id``, so robots cannot collide.
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless,
                 evaluation_settings=None, episode_sink=None, trajectory_sink=None, physics_sink=None):
        self.evaluation = (SimpleNamespace(**evaluation_settings)
                           if isinstance(evaluation_settings, dict)
                           else evaluation_settings or cfg.evaluation)
        self.metrics = None
        self.completed_episodes = []
        self.episode_sink = episode_sink
        self._evaluation_started = False
        self._episode_quota = int(getattr(self.evaluation, "episodes_per_env", 1))
        if self._episode_quota < 1:
            raise ValueError("episodes_per_env must be positive")
        self._seed = int(getattr(self.evaluation, "seed", 0))
        self._spawn_rngs = [np.random.default_rng(self._seed + i)
                            for i in range(cfg.env.num_envs)]
        self._validate_evaluation_cfg(cfg)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # Discard the constructor's unmeasured initialization draws.
        self._spawn_rngs = [np.random.default_rng(self._seed + i)
                            for i in range(self.num_envs)]
        self.episode_counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Raw-data attempt IDs advance even when an explicit reset interrupts a
        # trial; episode_counts alone tracks the completed-trial quota.
        self.attempt_indices = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._episode_status = [None] * self.num_envs
        self._episode_reason = [""] * self.num_envs
        self.measurement_joint_names = list(self.gym.get_actor_dof_names(self.envs[0], self.actor_handles[0]))
        if (self.measurement_joint_names != list(self.dof_names) or
                len(self.measurement_joint_names) != self.num_dof or
                self.num_actions != self.num_dof or self.torques.shape[1] != self.num_dof):
            raise ValueError("Recorded joint columns must match the actor DOF and applied torque ordering")
        self.metrics = VectorEpisodeMetrics(
            self.num_envs, self.device,
            trajectory_stride=int(getattr(self.evaluation, "trajectory_stride", 1)),
            trajectory_sink=trajectory_sink, physics_sink=physics_sink, num_joints=self.num_dof,
        )

    @staticmethod
    def _validate_evaluation_cfg(cfg):
        if cfg.terrain.mesh_type not in ("trimesh", "heightfield"):
            raise ValueError("Evaluation terrain requires trimesh or heightfield")
        if cfg.terrain.curriculum or cfg.commands.curriculum or cfg.commands.heading_command:
            raise ValueError("Evaluation requires fixed terrain and body-frame commands")
        if cfg.noise.add_noise or cfg.init_state.rand_init_dof:
            raise ValueError("Evaluation requires observation noise and initial DOF randomization off")
        if getattr(cfg.domain_rand, "randomize_torque_scale", True):
            raise ValueError("Evaluation must explicitly disable randomize_torque_scale")

    @property
    def evaluation_complete(self):
        return bool(self._evaluation_started and
                    torch.all(self.episode_counts >= self._episode_quota).item())

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        # Gym stores dt as float32: e.g. 0.005 becomes 0.004999999888.
        # Avoid an extra control step when a requested duration is an integral
        # number of nominal steps. Nonintegral durations still round upward.
        steps = self.max_episode_length_s / self.dt
        nearest = round(steps)
        self.max_episode_length = max(1, nearest if math.isclose(
            steps, nearest, rel_tol=1e-7, abs_tol=1e-7) else math.ceil(steps))

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id,
                                       self.physics_engine, self.sim_params)
        self.terrain = EvaluationTerrain(self.cfg.terrain, self.num_envs)
        if self.cfg.terrain.mesh_type == "trimesh":
            self._create_trimesh()
        else:
            self._create_heightfield()
        self._create_envs()

    def _get_env_origins(self):
        self.custom_origins = True
        self.terrain_types = torch.arange(self.num_envs, device=self.device) % len(self.terrain.type_names)
        self.terrain_levels = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.max_terrain_level = 1
        self.terrain_origins = torch.as_tensor(self.terrain.env_origins,
                                              dtype=torch.float, device=self.device)
        self.env_origins = self.terrain_origins[0, self.terrain_types].clone()
        self.traversal_bounds = torch.as_tensor(self.terrain.tile_bounds,
                                               dtype=torch.float, device=self.device)[self.terrain_types]
        bounds = self.terrain.tile_bounds
        self.terrain_x_min, self.terrain_x_max = float(bounds[:, 0].min()), float(bounds[:, 1].max())
        self.terrain_y_min, self.terrain_y_max = float(bounds[:, 2].min()), float(bounds[:, 3].max())
        self.goal_positions = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_directions = torch.zeros(self.num_envs, 2, device=self.device)
        self.spawn_positions = torch.zeros(self.num_envs, 3, device=self.device)

    def _process_rigid_body_props(self, props, env_id):
        props = super()._process_rigid_body_props(props, env_id)
        if not hasattr(self, "total_mass_kg"):
            self.total_mass_kg = torch.zeros(self.num_envs, device=self.device)
        # COT uses the whole articulated robot, not the base-body mass feature.
        self.total_mass_kg[env_id] = sum(float(body.mass) for body in props)
        return props

    def _reset_dofs(self, env_ids):
        # With dof_vel_use_pos_diff, self.dof_vel can be a derived tensor rather
        # than the simulator-state view. Clear the actual velocity before the
        # inherited method submits the DOF state, so no momentum survives reset.
        self.dof_state.view(self.num_envs, self.num_dof, 2)[env_ids, :, 1] = 0.
        super()._reset_dofs(env_ids)

    def _reset_root_states(self, env_ids):
        for env_id in env_ids.tolist():
            sample = self.terrain.sample_spawns(
                np.asarray([int(self.terrain_types[env_id])]), self._spawn_rngs[env_id],
                spawn_inset=float(getattr(self.evaluation, "spawn_inset", .75)),
                lateral_margin=float(getattr(self.evaluation, "lateral_margin", .75)),
                goal_inset=float(getattr(self.evaluation, "goal_inset", .75)),
            )
            ground = torch.as_tensor(sample["ground_positions"][0], device=self.device,
                                     dtype=self.root_states.dtype)
            direction = torch.as_tensor(sample["directions"][0], device=self.device,
                                        dtype=self.root_states.dtype)
            self.root_states[env_id] = self.base_init_state
            self.root_states[env_id, :3] = ground
            self.root_states[env_id, 2] += self.base_init_state[2]
            yaw = math.atan2(float(direction[1]), float(direction[0]))
            self.root_states[env_id, 3:7] = torch.tensor(
                [0., 0., math.sin(yaw / 2), math.cos(yaw / 2)], device=self.device)
            self.root_states[env_id, 7:13] = 0.
            self.spawn_positions[env_id] = self.root_states[env_id, :3]
            self.goal_positions[env_id] = torch.as_tensor(sample["goal_positions"][0], device=self.device)
            self.goal_directions[env_id] = direction
            self.traversal_bounds[env_id] = torch.as_tensor(sample["bounds"][0], device=self.device)
        indices = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(indices), len(indices))

    def _resample_commands(self, env_ids):
        self.commands[env_ids] = 0.
        self.commands[env_ids, 0] = float(getattr(self.evaluation, "command_vx", 1.0))

    def _post_physics_step_callback(self):
        # No command resampling, teleportation, curriculum, pushes, or randomizers.
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        self.base_height = (self.root_states[:, 2:3] - self.measured_heights).mean(dim=1)

    def step(self, actions, MINE=False):
        if not self._evaluation_started:
            raise RuntimeError("Call reset() before evaluation step()")
        self._step_commands = self.commands[:, :3].clone()
        actions = actions.clone()
        actions[self.episode_counts >= self._episode_quota] = 0.
        return super().step(actions, MINE=MINE)

    def _before_physics_substep(self):
        if self.metrics is not None and self._evaluation_started:
            # Read actual simulator velocity, regardless of the observation option
            # dof_vel_use_pos_diff. The applied torque already includes clipping/delay.
            self._omega_before = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1].clone()

    def _after_physics_substep(self):
        if self.metrics is not None and self._evaluation_started:
            omega_after = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
            self.metrics.physics_step(self.torques, self._omega_before, omega_after,
                                      self.sim_params.dt)

    def check_termination(self):
        # Preserve the training contact/tilt persistence rule, including its contact
        # counting, and add explicit evaluation-only crossing/boundary conditions.
        super().check_termination()
        failure = self.fail_buf > self.cfg.env.fail_to_terminal_time_s / self.dt
        max_tilt = float(getattr(self.evaluation, "max_tilt_deg", 60.0))
        tilted = self.projected_gravity[:, 2] > -math.cos(math.radians(max_tilt))
        position = self.root_states[:, :3]
        b = self.traversal_bounds
        outside = ((position[:, 0] < b[:, 0]) | (position[:, 0] > b[:, 1]) |
                   (position[:, 1] < b[:, 2]) | (position[:, 1] > b[:, 3]))
        invalid = ~torch.isfinite(self.root_states).all(dim=1)
        failure |= tilted | outside | invalid
        reached = ((position[:, :2] - self.goal_positions[:, :2]) * self.goal_directions).sum(dim=1) >= 0
        success = reached & ~failure
        timeout = (self.episode_length_buf >= self.max_episode_length) & ~failure & ~success
        self.reset_buf = failure | success | timeout
        self.time_out_buf = timeout
        if self.metrics is None or not self._evaluation_started:
            return
        self.metrics.control_step(
            position, self.root_states[:, 7:10], self.base_lin_vel,
            self.base_ang_vel, self._step_commands, self.dt,
            orientation_world=self.root_states[:, 3:7],
            angular_velocity_world=self.root_states[:, 10:13])
        for env_id in self.reset_buf.nonzero(as_tuple=False).flatten().tolist():
            if failure[env_id]:
                self._episode_status[env_id] = "failure"
                self._episode_reason[env_id] = ("nonfinite_state" if invalid[env_id] else
                                                "outside_terrain" if outside[env_id] else
                                                "tilt_limit" if tilted[env_id] else "training_failure")
            else:
                self._episode_status[env_id] = "success" if success[env_id] else "timeout"
                self._episode_reason[env_id] = "goal_crossed" if success[env_id] else "time_limit"

    def _finish(self, env_ids, status, reason):
        self.metrics.update_metadata(env_ids, [self._raw_state_metadata(env_id, "end")
                                               for env_id in env_ids.tolist()])
        rows = self.metrics.finish(env_ids, status, reason=reason)
        self.completed_episodes.extend(rows)
        if rows and self.episode_sink is not None:
            self.episode_sink(rows)
        if status != "interrupted":
            self.episode_counts[env_ids] += 1

    def _sync_reset_state(self, env_ids):
        self.base_pos[env_ids] = self.root_states[env_ids, :3]
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_lin_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 7:10])
        self.base_ang_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 10:13])
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])
        self.dof_acc[env_ids] = 0.
        self.torques[env_ids] = 0.
        self.last_dof_pos[env_ids] = self.dof_pos[env_ids]
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        self.base_height = (self.root_states[:, 2:3] - self.measured_heights).mean(dim=1)

    def _start(self, env_ids):
        if not len(env_ids):
            return
        self.attempt_indices[env_ids] += 1
        metadata = []
        for env_id in env_ids.tolist():
            self._episode_status[env_id] = None
            self._episode_reason[env_id] = ""
            metadata.append({
                "env_id": env_id, "episode_index": int(self.attempt_indices[env_id]),
                "terrain": self.terrain.type_names[int(self.terrain_types[env_id])],
                "terrain_id": int(self.terrain_types[env_id]),
                "seed": self._seed, "spawn_seed": self._seed + env_id,
                "eval_seed": self._seed,
                "task": getattr(self.evaluation, "task", ""),
                "checkpoint": str(getattr(self.evaluation, "checkpoint", "")),
                "spawn_position": self.spawn_positions[env_id].tolist(),
                "goal_position": self.goal_positions[env_id].tolist(),
                "command_vx": float(self.commands[env_id, 0]),
                "max_tilt_deg": float(getattr(self.evaluation, "max_tilt_deg", 60.0)),
                "training_failure_persistence_s": self.cfg.env.fail_to_terminal_time_s,
                **self._raw_episode_metadata(env_id),
                **self._raw_state_metadata(env_id, "start"),
            })
        gravity = self.sim_params.gravity
        gravity_magnitude = math.sqrt(gravity.x ** 2 + gravity.y ** 2 + gravity.z ** 2)
        self.metrics.start(
            env_ids, position_world=self.root_states[env_ids, :3],
            velocity_world=self.root_states[env_ids, 7:10],
            goal_direction_xy=self.goal_directions[env_ids], mass_kg=self.total_mass_kg[env_ids],
            gravity_m_s2=gravity_magnitude, metadata=metadata)

    def _raw_episode_metadata(self, env_id):
        gravity = self.sim_params.gravity
        vector = [float(gravity.x), float(gravity.y), float(gravity.z)]
        return {
            "dof_names": list(self.measurement_joint_names),
            "mass_kg": float(self.total_mass_kg[env_id]),
            "gravity_m_s2": math.sqrt(sum(value * value for value in vector)),
            "gravity_vector_m_s2": vector,
            "goal_direction_xy": self.goal_directions[env_id].detach().cpu().tolist(),
        }

    def _raw_state_metadata(self, env_id, prefix):
        def values(tensor):
            return [value if math.isfinite(value) else None for value in tensor.detach().cpu().tolist()]
        return {
            prefix + "_base_position": values(self.root_states[env_id, :3]),
            prefix + "_velocity_world": values(self.root_states[env_id, 7:10]),
            prefix + "_velocity_body": values(self.base_lin_vel[env_id]),
            prefix + "_angular_velocity_body": values(self.base_ang_vel[env_id]),
            prefix + "_angular_velocity_world": values(self.root_states[env_id, 10:13]),
            prefix + "_orientation_xyzw": values(self.root_states[env_id, 3:7]),
        }

    def reset_idx(self, env_ids):
        if not len(env_ids):
            return
        recording = self.metrics is not None and self._evaluation_started
        if recording:
            for env_id in env_ids.tolist():
                if bool(self.metrics.active[env_id]):
                    self._finish(env_ids.new_tensor([env_id]),
                                 self._episode_status[env_id] or "interrupted",
                                 self._episode_reason[env_id] or "explicit_reset")
        # Terminal metrics above refer to the physical endpoint, never the new spawn.
        super().reset_idx(env_ids)
        self._sync_reset_state(env_ids)
        if recording:
            pending = env_ids[self.episode_counts[env_ids] < self._episode_quota]
            self._start(pending)

    def reset(self):
        if self._evaluation_started:
            raise RuntimeError("Evaluation already started; automatic resets manage episode quotas")
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(env_ids)
        self._evaluation_started = True
        self._start(env_ids)
        self.compute_observations()
        clip = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clamp(self.obs_buf, -clip, clip)
        self.privileged_obs_buf = torch.clamp(self.privileged_obs_buf, -clip, clip)
        return self.obs_buf, self.privileged_obs_buf

    def interrupt(self, reason="interrupted"):
        """Flush unfinished traversals without reporting them as failures."""
        if self.metrics is not None:
            env_ids = self.metrics.active.nonzero(as_tuple=False).flatten()
            if len(env_ids):
                self._finish(env_ids, "interrupted", reason)
        return self.completed_episodes
