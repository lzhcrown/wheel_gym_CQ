# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
from .base_task import BaseTask
from wheel_legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, get_scale_shift
from wheel_legged_gym.utils.helpers import class_to_dict
from collections import deque
from .legged_robot_config import LeggedRobotCfg
import random
from .legged_robot import LeggedRobot
from wheel_legged_gym.utils.terrain import  Terrain
class Nezha2FreeEnv_PD(LeggedRobot):
    '''
    LuntuiFreeEnv is a class that represents a custom environment for a legged robot.

    Args:
        cfg (LeggedRobotCfg): Configuration object for the legged robot.
        sim_params: Parameters for the simulation.
        physics_engine: Physics engine used in the simulation.
        sim_device: Device used for the simulation.
        headless: Flag indicating whether the simulation should be run in headless mode.

    Attributes:
        sim (gymtorch.GymSim): The simulation object.
        terrain (HumanoidTerrain): The terrain object.
        up_axis_idx (int): The index representing the up axis.
        command_input (torch.Tensor): Tensor representing the command input.
        privileged_obs_buf (torch.Tensor): Tensor representing the privileged observations buffer.
        obs_buf (torch.Tensor): Tensor representing the observations buffer.
        obs_history (collections.deque): Deque containing the history of observations.
        critic_history (collections.deque): Deque containing the history of critic observations.

    Methods:
        _push_robots(): Randomly pushes the robots by setting a randomized base velocity.
        create_sim(): Creates the simulation, terrain, and environments.
        _get_noise_scale_vec(cfg): Sets a vector used to scale the noise added to the observations.
        step(actions): Performs a simulation step with the given actions.
        compute_observations(): Computes the observations.
        reset_idx(env_ids): Resets the environment for the specified environment IDs.
    '''
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05   # 0.05
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.ref_dof_vel = torch.zeros_like(self.dof_pos)  
        self.ref_dof_pos = self.default_dof_pos.expand(self.num_envs, -1).clone()



    def  _get_phase(self):
        cycle_time = 4
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase
    
    def compute_ref_state(self):
        """ 参考轨迹v0,但参考轨迹的foot是始终在中线上的,如果权重太大,会导致迈不开腿
        """
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # sin_pos_2 = torch.sin(2 * torch.pi * phase + torch.pi)
        sin_pos_p = sin_pos.clone()
        # sin_pos_n = sin_pos.clone()
        # sin_pos_2_p = sin_pos_2.clone()
        # sin_pos_2_n = sin_pos_2.clone()
        self.ref_dof_vel = torch.zeros_like(self.dof_pos)   # 错误：ref_dof_pos不应该为0， 而是应该等于default joint pos
        self.ref_dof_pos = self.default_dof_pos.expand(self.num_envs, -1).clone()
        # print('dim', self.ref_dof_pos.size())
        # left foot stance phase set to default joint pos
        sin_pos_p[sin_pos_p < 0] = 0
        # sin_pos_n[sin_pos_n > 0] = 0
        # sin_pos_2_p[sin_pos_2_p < 0] = 0
        # sin_pos_2_n[sin_pos_2_n > 0] = 0
        self.ref_dof_pos[:, 0] += sin_pos_p * 0.25 #FL_hip
        self.ref_dof_pos[:, 1] += sin_pos_p * 0.3  #FL_thigh                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             #FL_thigh
        self.ref_dof_pos[:, 2] -= sin_pos_p * 0.3 #FL_calf
        self.ref_dof_vel[:, 3] += sin_pos * 5     #FL_foot

        self.ref_dof_pos[:, 4] -= sin_pos_p * 0.25 #FR_hip
        self.ref_dof_pos[:, 5] -= sin_pos_p * 0.3 #FR_thigh
        self.ref_dof_pos[:, 6] += sin_pos_p * 0.3 #FR_calf
        self.ref_dof_vel[:, 7] += sin_pos * 5     #FR_foot

        self.ref_dof_pos[:, 8] += sin_pos_p * 0.25 #RL_hip
        self.ref_dof_pos[:, 9] += sin_pos_p * 0.3 #RL_thigh
        self.ref_dof_pos[:, 10] -= sin_pos_p * 0.3#RL_calf
        self.ref_dof_vel[:, 11] += sin_pos * 5    #RL_foot

        self.ref_dof_pos[:, 12] -= sin_pos_p * 0.25#RR_hip
        self.ref_dof_pos[:, 13] += sin_pos_p * 0.3#RR_thigh
        self.ref_dof_pos[:, 14] -= sin_pos_p * 0.3#RR_calf
        self.ref_dof_vel[:, 15] += sin_pos * 5    #RR_foot
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pos_ref = actions * self.cfg.control.pos_action_scale
        # pos_ref[:, 1] *= 0
        # pos_ref[:, 3] *= 0
        # vel_ref = actions * self.cfg.control.vel_action_scale
        # vel_ref[:, :1] *= 0
        # vel_ref[:, 2:3] *= 0
        # # pd controller
        # self.joint_pos_target = pos_ref + self.default_dof_pos

        # if self.cfg.domain_rand.randomize_PD_factor:
        #     p_gains = self.Kp_factors * self.p_gains
        #     d_gains = self.Kd_factors * self.d_gains
        # else:
        #     p_gains = self.p_gains
        #     d_gains = self.d_gains

        # if self.cfg.domain_rand.randomize_coulomb_friction:
        #     torques = p_gains * (self.joint_pos_target - self.dof_pos + self.motor_offsets) +\
        #     d_gains * (vel_ref - self.dof_vel) -\
        #     self.randomized_joint_viscous * self.dof_vel -\
        #     self.randomized_joint_coulomb * torch.sign(self.dof_vel)
        # else:
        self.compute_ref_state()
        p_gains = self.p_gains
        d_gains = self.d_gains
        # print(self.dof_vel.shape)
        # print(self.dof_vel[0,[3,7,11,15]])
        torques = p_gains * (self.ref_dof_pos - self.dof_pos + self.motor_offsets) + d_gains * (self.ref_dof_vel - self.dof_vel)
        if self.cfg.domain_rand.randomize_motor_strength:
            torques *= self.motor_strengths

        # T1, T2 = self.compute_motor_torque(
        #     200, 0.0
        # )

        # current_time = ti.time()
        # if current_time - self.last_print_time >= 0.2:
        #     self.last_print_time = current_time
        #     print("results---------------------------")
        #     print("torques", torques[0,:])
        #     # print("T1", T1[0,:])
        #     # print("T2", T2[0,:])
        #     # print("left", (pos_ref[0,2] - self.dof_vel[0,2]))
        #     # print("right", (pos_ref[0,5] - self.dof_vel[0,5]))
        #     # print("left", (vel_ref[0,2] - self.dof_vel[0,2]))
        #     # print("right", (vel_ref[0,5] - self.dof_vel[0,5]))
        #     print('left ref vel', vel_ref[0,2])
        #     print('left real vel', self.dof_vel[0,2])
        #     print('right ref vel', vel_ref[0,5])
        #     print('right real vel', self.dof_vel[0,5])
        #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        # # 检查是否是nan
        # if torch.isnan(T1).any() & torch.isnan(T2).any():
        #     # print("torques is nan")
        #     T1 = self.last_T1
        #     T2 = self.last_T2

        # torques[:,0] -= T1[:, 0]
        # torques[:,2] -= T1[:, 1]
        # torques[:,1] -= T2[:, 0]
        # torques[:,4] -= T2[:, 1]
        # self.last_T1 = T1
        # self.last_T2 = T2

        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    # TODO
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        #lagged_base_ang_vel
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        #base_euler_xyz
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        #projected_gravity
        noise_vec[6:9] = noise_scales.gravity * noise_level
        #commands
        noise_vec[9:12] = 0.
        #q
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        #dq
        noise_vec[24:40] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[40:] = 0
        return noise_vec


    def step(self, actions, KEYBOARD = False, HIM = False):
        if self.cfg.env.use_ref_actions:
            actions += self.ref_action
        if self.cfg.control.action_smoothness:
            ratio = self.cfg.control.ratio
            self.actions = ratio * self.actions + (1 - ratio) * self.last_actions  # debug plot    
        # dynamic randomization
        delay = torch.rand((self.num_envs, 1), device=self.device) * self.cfg.domain_rand.action_delay

        actions = (1 - delay) * actions + delay * self.actions
        actions += self.cfg.domain_rand.action_noise * torch.randn_like(actions) * actions

        return super().step(actions, KEYBOARD = KEYBOARD, HIM = HIM)
    

    def compute_privileged_observations(self):
        # compute the critic_obs ,not change critic_history

        self.base_height_obs = self.base_height.unsqueeze(1)
        heights = (
            torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1,
                1.0,
            )
            * self.obs_scales.height_measurements
        )
        contact_forces_scale, contact_forces_shift = get_scale_shift(self.cfg.normalization.contact_force_range)
        privileged_termination_obs = torch.cat(( # 3+12+16+16+3+3+16+16+1+3+16+16+16
            self.commands[:, :3] * self.commands_scale,  # 3
            (self.dof_pos[:, [0,1,2,5,6,7,9,10,11,13,14,15]] - self.default_dof_pos[:, [0,1,2,5,6,7,9,10,11,13,14,15]]) * self.obs_scales.dof_pos,  # 12 
            self.dof_vel * self.obs_scales.dof_vel,  # 16
            self.actions,  # 16
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz * self.obs_scales.quat,  # 3
            # self.projected_gravity, # 重力投影
            self.dof_acc * self.obs_scales.dof_acc, # 16
            self.torques * self.obs_scales.torques, # 16
            (self.body_mass - self.body_mass.mean()).view(self.num_envs, 1),    # 1
            self.base_com, # 3
            self.default_dof_pos - self.raw_default_dof_pos, #16 
            # self.default_dof_pos
            self.p_gains / 20.0, # 16
            self.d_gains / 0.2, # 16
            # self.ref_dof_pos[:, [0,1,3,4]] - self.dof_pos[:, [0,1,3,4]],
            # self.ref_dof_vel[:, [2,5]] - self.dof_vel[:, [2,5]],
            # sin_pos_obs,
        ), dim=-1)
        self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
                                            (self.contact_forces.view(self.num_envs, -1) - contact_forces_shift) * contact_forces_scale), dim=-1)
        privileged_termination_obs = torch.cat((privileged_termination_obs,
                                            heights), dim=1)
        privileged_termination_obs = torch.cat((privileged_termination_obs, 
                                                self.base_lin_vel * self.obs_scales.lin_vel,
                                            self.base_height_obs * self.obs_scales.height_measurements), dim=1)
        if self.cfg.env.c_frame_stack > 1:           
            return torch.cat([torch.cat([self.critic_history[i+1] for i in range(self.cfg.env.c_frame_stack-1)], dim=1),
                              privileged_termination_obs], dim=1)
        return privileged_termination_obs



    # TODO
    def compute_observations(self):
        # phase = self._get_phase()
        # sin_pos_obs = torch.sin(2*torch.pi*phase).unsqueeze(1)
        # self.compute_ref_state()
        self.base_height_obs = self.base_height.unsqueeze(1)
        heights = (
            torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1,
                1.0,
            )
            * self.obs_scales.height_measurements
        )
        contact_forces_scale, contact_forces_shift = get_scale_shift(self.cfg.normalization.contact_force_range)
        self.privileged_obs_buf = torch.cat(( #3+12+16+16+3+3+3+16+16+1+3+16+16+16+51+187+4
            self.commands[:, :3] * self.commands_scale,  # 3
            (self.dof_pos[:, [0,1,2,5,6,7,9,10,11,13,14,15]] - self.default_dof_pos[:, [0,1,2,5,6,7,9,10,11,13,14,15]]) * self.obs_scales.dof_pos,  # 12 
            self.dof_vel * self.obs_scales.dof_vel,  # 16
            self.actions,  # 16
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz * self.obs_scales.quat,  # 3
            self.projected_gravity, # 重力投影3
            self.dof_acc * self.obs_scales.dof_acc, # 16
            self.torques * self.obs_scales.torques, # 16
            (self.body_mass - self.body_mass.mean()).view(self.num_envs, 1),    # 1
            self.base_com, # 3
            self.default_dof_pos - self.raw_default_dof_pos, # 16
            # self.default_dof_pos
            self.p_gains / 20.0, # 16
            self.d_gains / 0.2, # 16
            # self.ref_dof_pos[:, [0,1,3,4]] - self.dof_pos[:, [0,1,3,4]],
            # self.ref_dof_vel[:, [2,5]] - self.dof_vel[:, [2,5]],
            # sin_pos_obs,
        ), dim=-1)
        self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,#17*3
                                            (self.contact_forces.view(self.num_envs, -1) - contact_forces_shift) * contact_forces_scale), dim=-1)
        # 加入额外的privileged
        # if self.cfg.env.priv_observe_friction:
        #      # 1
        #     friction_coeffs_scale, friction_coeffs_shift = get_scale_shift(self.cfg.domain_rand.friction_range)
        #     self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
        #                                             (self.friction_coeffs[:, 0].unsqueeze(1)
        #                                             - friction_coeffs_shift) * friction_coeffs_scale),dim=1)
        
        # if self.cfg.env.priv_observe_restitution:
        #     # 1
        #     restitutions_scale, restitutions_shift = get_scale_shift(self.cfg.domain_rand.restitution_range)
        #     self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
        #                                          (self.restitutions[:, 0].unsqueeze(1) 
        #                                          - restitutions_shift) * restitutions_scale),dim=1)

        # if self.cfg.env.priv_observe_base_mass:
        #     # 1
        #     payloads_scale, payloads_shift = get_scale_shift(self.cfg.domain_rand.added_mass_range)
        #     self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
        #                                          (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale),dim=1)

        # if self.cfg.env.priv_observe_com_displacement:
        #     # 3
        #     com_displacements_scale, com_displacements_shift = get_scale_shift(
        #         self.cfg.domain_rand.com_displacement_range)
        #     self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
        #                                          (self.com_displacements - com_displacements_shift) * com_displacements_scale), dim=1)
        # if self.cfg.env.priv_observe_motor_strength:
        #     # 6
        #     motor_strengths_scale, motor_strengths_shift = get_scale_shift(self.cfg.domain_rand.motor_strength_range)
        #     self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
        #                                          (self.motor_strengths - motor_strengths_shift) * motor_strengths_scale), dim=1)
 
        # if self.cfg.env.priv_observe_motor_offset:
        #     # 6
        #     motor_offset_scale, motor_offset_shift = get_scale_shift(self.cfg.domain_rand.motor_offset_range)
        #     self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
        #                                          (self.motor_offsets - motor_offset_shift) * motor_offset_scale), dim=1) 
        # if self.cfg.env.priv_observe_gravity:
        #     # 3
        #     gravity_scale, gravity_shift = get_scale_shift(self.cfg.domain_rand.gravity_range)
        #     self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,
        #                                          (self.gravities - gravity_shift) / gravity_scale), dim=1)

        # 始终将需要预测的值量放在最后（若有预测网络）
        self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,#187
                                            heights), dim=1) 
        self.privileged_obs_buf = torch.cat((self.privileged_obs_buf,#4
                                            self.base_lin_vel * self.obs_scales.lin_vel, 
                                            self.base_height_obs * self.obs_scales.height_measurements), dim=1)
        if self.cfg.domain_rand.add_dof_lag:
            if self.cfg.domain_rand.randomize_dof_lag_timesteps_perstep:
                self.dof_lag_timestep = torch.randint(self.cfg.domain_rand.dof_lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.dof_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.dof_lag_timestep > self.last_dof_lag_timestep + 1
                self.dof_lag_timestep[cond] = self.last_dof_lag_timestep[cond] + 1
                self.last_dof_lag_timestep = self.dof_lag_timestep.clone()
            self.lagged_dof_pos = self.dof_lag_buffer[torch.arange(self.num_envs), :, self.dof_lag_timestep.long()]
            self.lagged_dof_vel = self.dof_vel_lag_buffer[torch.arange(self.num_envs), :, self.dof_lag_timestep.long()]
        else:
            self.lagged_dof_pos = self.dof_pos
            self.lagged_dof_vel = self.dof_vel

        if self.cfg.domain_rand.add_imu_lag:
            if self.cfg.domain_rand.randomize_imu_lag_timesteps_perstep:
                self.imu_lag_timestep = torch.randint(self.cfg.domain_rand.imu_lag_timesteps_range[0], 
                                                  self.cfg.domain_rand.imu_lag_timesteps_range[1]+1,(self.num_envs,),device=self.device)
                cond = self.imu_lag_timestep > self.last_imu_lag_timestep + 1
                self.imu_lag_timestep[cond] = self.last_imu_lag_timestep[cond] + 1
                self.last_imu_lag_timestep = self.imu_lag_timestep.clone()
            self.lagged_imu = self.imu_lag_buffer[torch.arange(self.num_envs), :, self.imu_lag_timestep.long()]
            self.lagged_base_ang_vel = self.lagged_imu[:,:3].clone()
            self.lagged_base_euler_xyz = self.lagged_imu[:,-3:].clone()
            self.lagged_projected_gravity = self.lagged_imu[:,-3:].clone()
        # no imu lag
        else:              
            self.lagged_base_ang_vel = self.base_ang_vel[:,:3]
            self.lagged_base_euler_xyz = self.base_euler_xyz
            self.lagged_projected_gravity = self.projected_gravity
        q = (self.lagged_dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.lagged_dof_vel * self.obs_scales.dof_vel

        obs_buf = torch.cat((#3+3+3+3+12+16+16
            self.lagged_base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.lagged_base_euler_xyz * self.obs_scales.quat,  # 3
            self.lagged_projected_gravity,#3
            self.commands[:, :3]  * self.commands_scale,   # 3 
            q[:, [0,1,2,5,6,7,9,10,11,13,14,15]], #12
            dq,  # 16
            self.actions,   # 16
            
        ), dim=-1)
        obs_now = obs_buf.clone()

        if self.cfg.domain_rand.randomize_obs_delay:    
            obs_now = self.hist_obs.popleft().to(self.device)
            self.hist_obs.append(obs_buf)

        if self.add_noise:  
            add_noise = torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
            obs_now += add_noise

        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)
        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)
    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(WHEEL_LEGGED_GYM_ROOT_DIR=WHEEL_LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        num_joints = self.gym.get_asset_joint_count(robot_asset)
        print("Number of joints in the asset:", num_joints)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        # knee_names = [s for s in body_names if self.cfg.asset.knee_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
        wheel_names =[]
        for name in self.cfg.asset.wheel_name:
            wheel_names.extend([s for s in self.dof_names if name in s])
        print("###self.rigid_body names:",body_names)
        print("###self.dof names:",self.dof_names)
        print("###penalized_contact_names:",penalized_contact_names)
        print("###termination_contact_names:",termination_contact_names)
        print("###feet_names:",feet_names)
        print("###wheels name:",wheel_names)
        
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.base_com = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.body_mass = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device, requires_grad=False)
        # xxx added_rand
        self._init_custom_buffers__()
        self._randomize_dof_props(torch.arange(self.num_envs, device=self.device))
        self._randomize_gravity()
        self._randomize_rigid_body_props(torch.arange(self.num_envs, device=self.device), self.cfg)
        # xxx

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
                        
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self._refresh_actor_dof_props(torch.arange(self.num_envs, device=self.device))  # dof random
        dof_props = self.gym.get_actor_dof_properties(self.envs[0], 0)
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
        # self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        # for i in range(len(knee_names)):
        #     self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])
            

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
            print(f"##### the {i} rigid body:{termination_contact_names[i]}=={self.termination_contact_indices[i]}")
        self.wheel_indices = torch.zeros(len(wheel_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(wheel_names)):
            self.wheel_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], wheel_names[i])

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

# ================================================ Rewards Humanoid Gym ================================================== #
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        self.dof_vel[:,self.wheel_indices] = 0
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        self.dof_vel[:,self.wheel_indices] = 0
        self.last_dof_vel[:,self.wheel_indices] = 0
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_feet_contact_uniform(self):
    # 1. 获取接触状态 (1表示接触, 0表示悬空)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
    
    # 2. 计算每个环境下，4条腿的平均接触率
        contact_mean = torch.mean(contact.float(), dim=1)
    
    # 3. 计算每条腿接触率与平均值的差的绝对值
    # 如果某条腿一直不接触(0)，而平均值是0.5，diff就是0.5
        contact_diff = torch.abs(contact.float() - contact_mean.unsqueeze(1))
    
    # 4. 汇总惩罚：差异越大，惩罚越大
        rew_uniform = -torch.sum(contact_diff, dim=1)
        return rew_uniform


    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands        
        dof_err = self.dof_pos - self.default_dof_pos
        dof_err[:,self.wheel_indices] = 0
        return torch.sum(torch.abs(dof_err), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    def _reward_orientation_quat(self):
        # Penalize non flat base orientation
        
        orientation_error = torch.sum(torch.square(self.root_states[:,:7] - self.base_init_state[0:7]),dim=1)
        return torch.exp(-orientation_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_hip_action_l2(self):
        action_l2 = torch.sum(self.actions[:, [0, 4, 8, 12]] ** 2, dim=1)
        #self.episode_metric_sums['leg_action_l2'] += action_l2
        return action_l2
    
    def _reward_hip_default(self):
        hip_err = torch.sum((self.dof_pos[:, [0, 4, 8, 12]] - self.default_dof_pos[:, [0, 4, 8, 12]]) ** 2, dim = 1)
        # print("penalty",penalty.shape)
        return hip_err
    def _reward_run_still(self):
        # Penalize motion at running commands        
        dof_err = self.dof_pos - self.default_dof_pos
        dof_err[:,self.wheel_indices] = 0
        return torch.sum(torch.abs(dof_err), dim=1) * (torch.norm(self.commands[:, :2], dim=1) > 0.1)
    def _reward_action_symmetry(self):
        # 假设动作向量的排列是 [FL_hip, FL_knee, FR_hip, FR_knee, ...]
        # 即偶数索引是左腿，奇数索引是右腿（具体取决于你的动作空间定义）
        
        # 提取左腿动作 (所有偶数索引)
        thigh_actions = self.actions[:, [1, 5, 9, 13]]
        
        # 提取右腿动作 (所有奇数索引)
        calf_actions = self.actions[:, [2, 6, 10, 14]]
        
        # 计算左右腿动作的差值
        diff = left_leg_actions - right_leg_actions
        
        # 计算均方误差
        mse = torch.mean(diff**2, dim=1)
        
        # 转化为奖励：差异越小，奖励越高
        return torch.exp(-10.0 * mse)

    