# 标定仿真器friction、damping、amature的play文件

# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 标定仿真器friction、damping、amature的play文件
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

import os
import cv2
import numpy as np
from isaacgym import gymapi
from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR
import time
# import isaacgym
from wheel_legged_gym.envs import *
from wheel_legged_gym.utils import  get_args, export_policy_as_jit, task_registry
from wheel_legged_gym.utils import Logger_pd
from isaacgym.torch_utils import *
import torch.nn.functional as F

import torch
from tqdm import tqdm
from datetime import datetime
import json
import copy

class Integrated_RMA_policy(torch.nn.Module):
    def __init__(self, actor, adaptation_encoder, num_single_obs, frame_stack, actor_input_stack):
        super().__init__()
        self.actor = copy.deepcopy(actor).cpu()
        self.adaptation_encoder = copy.deepcopy(adaptation_encoder).cpu()
        self.num_single_obs = num_single_obs
        self.actor_input_stack = actor_input_stack
        self.frame_stack = frame_stack
    
    def forward(self, obs):
        latent = self.adaptation_encoder.forward(obs)
        action_mean = self.actor.forward(torch.cat([obs[:, (self.frame_stack - self.actor_input_stack) * self.num_single_obs:], latent], dim=-1))
        return action_mean

    def export(self, path):
        path = os.path.join(path, "policy_rma.pt")
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

class Integrated_EST_policy(torch.nn.Module):
    def __init__(self, actor, estimator, num_single_obs, frame_stack, actor_input_stack):
        super().__init__()
        self.actor = copy.deepcopy(actor).cpu()
        self.estimator = copy.deepcopy(estimator).cpu()
        self.num_single_obs = num_single_obs
        self.actor_input_stack = actor_input_stack
        self.frame_stack = frame_stack
    
    def forward(self, obs):
        latent = self.estimator.forward(obs)
        action_mean = self.actor.forward(torch.cat([obs[:, (self.frame_stack - self.actor_input_stack) * self.num_single_obs:], latent], dim=-1))
        output = torch.cat((action_mean, latent), dim=-1)
        return output

    def export(self, path):
        path = os.path.join(path, "policy_est.pt")
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path) 

class Integrated_MINE_policy(torch.nn.Module):
    def __init__(self, actor, encoder, num_single_obs, frame_stack):
        super().__init__()
        self.actor = copy.deepcopy(actor).cpu()
        self.encoder = copy.deepcopy(encoder).cpu()
        self.num_single_obs = num_single_obs
        self.frame_stack = frame_stack
    
    def forward(self, obs):
        parts = self.encoder.forward(obs)
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2.0)
        action_mean = self.actor.forward(torch.cat([obs[:,-self.num_single_obs:], vel, z], dim=-1))
        output = torch.cat((action_mean, vel), dim=-1)
        return output

    def export(self, path):
        path = os.path.join(path, "policy_mine.pt")
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path) 

class PolicyExporterMINE(torch.nn.Module):
    def __init__(self, actor_critic, num_single_obs):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.estimator = copy.deepcopy(actor_critic.estimator.encoder)
        self.num_single_obs = num_single_obs 

    def forward(self, obs_history):
        parts = self.estimator(obs_history)[:, 0:19]
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2.0)
        return self.actor(torch.cat((obs_history[:, -self.num_single_obs:], vel, z), dim=1))

    def export(self, path):
        path = os.path.join(path, 'policy_mine_export.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

def play(args):


    args.run_name = args.task
    print('args.task-------------------------',args.task)
    
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.seed = 10
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)
    env_cfg.sim.max_gpu_contact_pairs = 2**10
    # env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False     
    env_cfg.terrain.max_init_terrain_level = 5
    
    
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(WHEEL_LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'polices')
        if not os.path.exists(path):
            os.makedirs(path)
        if args.task == "luntui_rma":
            print("export rma policy",env_cfg.env.num_single_obs, env_cfg.env.actor_input_stack, env_cfg.env.frame_stack)
            integrated_policy = Integrated_RMA_policy(ppo_runner.alg.actor_critic.actor,
                                                ppo_runner.alg.actor_critic.adaptation_encoder,
                                                env_cfg.env.num_single_obs, env_cfg.env.frame_stack, env_cfg.env.actor_input_stack)
            integrated_policy.export(path)
        elif args.task == "luntui_est":
            print("export est policy",env_cfg.env.num_single_obs, env_cfg.env.actor_input_stack, env_cfg.env.frame_stack)
            integrated_policy = Integrated_EST_policy(ppo_runner.alg.actor_critic.actor,
                                                ppo_runner.alg.actor_critic.estimator,
                                                env_cfg.env.num_single_obs, env_cfg.env.frame_stack, env_cfg.env.actor_input_stack)
            integrated_policy.export(path)
        elif args.task == "luntui_mine":
            print("export mine policy",env_cfg.env.num_single_obs, env_cfg.env.c_frame_stack, env_cfg.env.frame_stack, train_cfg.policy.enc_hidden_dims[-1])
            integrated_policy = Integrated_MINE_policy(ppo_runner.alg.actor_critic.actor,
                                                ppo_runner.alg.actor_critic.estimator.encoder,
                                                env_cfg.env.num_single_obs, env_cfg.env.frame_stack)
            integrated_policy.export(path)
            export_policy_mine = PolicyExporterMINE(ppo_runner.alg.actor_critic, env_cfg.env.num_single_obs)
            export_policy_mine.export(path) # 源代码中导出模型的方法
        else:
            export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)


    logger_pd = Logger_pd(env.dt)
    robot_index = 0 # which robot is used for logging
    stop_state_log = 600 # number of steps before plotting states    FL_hip_pos_list = []
    
    FL_hip_pos_list = []
    FL_thigh_pos_list = []
    FL_calf_pos_list = []
    FL_wheel_pos_list = []
    FL_hip_vel_list = []
    FL_thigh_vel_list = []
    FL_calf_vel_list = []
    FL_wheel_vel_list = []
    FL_hip_torque_list = []
    FL_thigh_torque_list = []
    FL_calf_torque_list = []
    FL_wheel_torque_list = []
    
    FR_hip_pos_list = []
    FR_thigh_pos_list = []
    FR_calf_pos_list = []
    FR_wheel_pos_list = []
    FR_hip_vel_list = []
    FR_thigh_vel_list = []
    FR_calf_vel_list = []
    FR_wheel_vel_list = []
    FR_hip_torque_list = []
    FR_thigh_torque_list = []
    FR_calf_torque_list = []
    FR_wheel_torque_list = []

    RL_hip_pos_list = []
    RL_thigh_pos_list = []
    RL_calf_pos_list = []
    RL_wheel_pos_list = []
    RL_hip_vel_list = []
    RL_thigh_vel_list = []
    RL_calf_vel_list = []
    RL_wheel_vel_list = []
    RL_hip_torque_list = []
    RL_thigh_torque_list = []
    RL_calf_torque_list = []
    RL_wheel_torque_list = []

    RR_hip_pos_list = []
    RR_thigh_pos_list = []
    RR_calf_pos_list = []
    RR_wheel_pos_list = []
    RR_hip_vel_list = []
    RR_thigh_vel_list = []
    RR_calf_vel_list = []
    RR_wheel_vel_list = []
    RR_hip_torque_list = []
    RR_thigh_torque_list = []
    RR_calf_torque_list = []
    RR_wheel_torque_list = []

    FL_hip_pos_ref_list = []
    FL_thigh_pos_ref_list = []
    FL_calf_pos_ref_list = []
    FL_wheel_vel_ref_list = []

        
    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        camera_offset = gymapi.Vec3(1, -1, 0.5)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
                                                    np.deg2rad(135))
        actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
        env.gym.attach_camera_to_body(
            h1, env.envs[0], body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_dir = os.path.join(WHEEL_LEGGED_GYM_ROOT_DIR, 'videos')
        experiment_dir = os.path.join(WHEEL_LEGGED_GYM_ROOT_DIR, 'videos', train_cfg.runner.experiment_name)
        dir = os.path.join(experiment_dir, datetime.now().strftime('%b%d_%H-%M-%S')+ args.run_name + '.mp4')
        if not os.path.exists(video_dir):
            os.mkdir(video_dir)
        if not os.path.exists(experiment_dir):
            os.mkdir(experiment_dir)
        video = cv2.VideoWriter(dir, fourcc, 100.0, (1920, 1080))

    motor_vel_list = []
    ang_vel_list = []
    feet_height_list = []
    air_time_list = []
    feet_height_smooth_list = []
    vel_list = [[] for i in range(3)]
    count = 0

    CoM_offset_compensate = True
    vel_err_intergral = torch.zeros(env.num_envs, device=env.device)
    vel_cmd = torch.zeros(env.num_envs, device=env.device)
    command = [0.5, 0, 0]
    # for i in tqdm(range(stop_state_log)):
    for i in range(stop_state_log):
        if i > stop_state_log - 4 and i < stop_state_log - 1:
            logger_pd.plot_states()  # 绘图
        if args.task == "luntui_rma":
            actions = policy(obs.detach()[:, (env_cfg.env.frame_stack - env_cfg.env.actor_input_stack) * env_cfg.env.num_single_obs:], obs.detach())
        elif args.task == "luntui_est" : #or args.task == "luntui_plane"
            actions = policy(obs.detach()[:, (env_cfg.env.frame_stack - env_cfg.env.actor_input_stack) * env_cfg.env.num_single_obs:], obs.detach())
        else :
            #actions = policy(obs.detach()[:, (env_cfg.env.frame_stack - env_cfg.env.actor_input_stack) * env_cfg.env.num_single_obs:], obs.detach())
            actions = policy(obs.detach()) 
            
            
            # pred_action = jit_model(obs.detach())
            # actions = pred_action[:, :6]

            # print(obs.detach().shape)

        if FIX_COMMAND:
            env.commands[:, 0] = 1. # 1.0
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0
        else:
            env.commands[:, 0] = 1.
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0

            if CoM_offset_compensate:
                if i > 400 and i < 800:
                    vel_cmd[:] = env.commands[:, 0] #* np.clip((i - 200) / 400.0, 0, 1)
                # elif i>= 800 and i < 1400:
                #     vel_cmd[:] = env.commands[:, 0] * np.clip((1400 - i) / 600.0, 0, 1)
                else:
                    vel_cmd[:] = 0
                # vel_err_intergral += (
                #     (vel_cmd - env.base_lin_vel[:, 0])
                #     * env.dt
                #     * ((vel_cmd - env.base_lin_vel[:, 0]).abs() < 0.5)
                # )
                # vel_err_intergral = torch.clip(vel_err_intergral, -0.5, 0.5)
                env.commands[:, 0] = vel_cmd # + vel_err_intergral
        # print("vel_x:", env.base_lin_vel[:,0])

        if args.task=="luntui_mine":
            obs, critic_obs, rews, dones, infos, _, __ , command= env.step(actions.detach(), KEYBOARD = True, MINE = True)
        else:
            obs, critic_obs, rews, dones, infos = env.step(actions.detach())
        env.gym.refresh_dof_state_tensor(env.sim)#更新tensor，不然没有轮子的速度数值
        # print(env.lag_buffer)

        # print(env.dof_vel[robot_index, [3,7,11,15]])
        FL_hip_pos_list.append(env.dof_pos[:, 0].cpu().detach().numpy().tolist()[robot_index])    
        FL_thigh_pos_list.append(env.dof_pos[:, 1].cpu().detach().numpy().tolist()[robot_index])
        FL_calf_pos_list.append(env.dof_pos[:, 2].cpu().detach().numpy().tolist()[robot_index])
        FL_wheel_pos_list.append(env.dof_pos[:, 3].cpu().detach().numpy().tolist()[robot_index])
        
        FL_hip_vel_list.append(env.dof_vel[:, 0].cpu().detach().numpy().tolist()[robot_index])
        FL_thigh_vel_list.append(env.dof_vel[:, 1].cpu().detach().numpy().tolist()[robot_index])
        FL_calf_vel_list.append(env.dof_vel[:, 2].cpu().detach().numpy().tolist()[robot_index])
        FL_wheel_vel_list.append(env.dof_vel[:, 3].cpu().detach().numpy().tolist()[robot_index])
        
        FL_hip_torque_list.append(env.torques[:, 0].cpu().detach().numpy().tolist()[robot_index])
        FL_thigh_torque_list.append(env.torques[:, 1].cpu().detach().numpy().tolist()[robot_index])
        FL_calf_torque_list.append(env.torques[:, 2].cpu().detach().numpy().tolist()[robot_index])
        FL_wheel_torque_list.append(env.torques[:, 3].cpu().detach().numpy().tolist()[robot_index])
        
        FR_hip_pos_list.append(env.dof_pos[:, 0].cpu().detach().numpy().tolist()[robot_index])    
        FR_thigh_pos_list.append(env.dof_pos[:, 1].cpu().detach().numpy().tolist()[robot_index])
        FR_calf_pos_list.append(env.dof_pos[:, 2].cpu().detach().numpy().tolist()[robot_index])
        FR_wheel_pos_list.append(env.dof_pos[:, 3].cpu().detach().numpy().tolist()[robot_index])
        
        FR_hip_vel_list.append(env.dof_vel[:, 0].cpu().detach().numpy().tolist()[robot_index])
        FR_thigh_vel_list.append(env.dof_vel[:, 1].cpu().detach().numpy().tolist()[robot_index])
        FR_calf_vel_list.append(env.dof_vel[:, 2].cpu().detach().numpy().tolist()[robot_index])
        FR_wheel_vel_list.append(env.dof_vel[:, 3].cpu().detach().numpy().tolist()[robot_index])
        
        FR_hip_torque_list.append(env.torques[:, 0].cpu().detach().numpy().tolist()[robot_index])
        FR_thigh_torque_list.append(env.torques[:, 1].cpu().detach().numpy().tolist()[robot_index])
        FR_calf_torque_list.append(env.torques[:, 2].cpu().detach().numpy().tolist()[robot_index])
        FR_wheel_torque_list.append(env.torques[:, 3].cpu().detach().numpy().tolist()[robot_index])

        RL_hip_pos_list.append(env.dof_pos[:, 0].cpu().detach().numpy().tolist()[robot_index])    
        RL_thigh_pos_list.append(env.dof_pos[:, 1].cpu().detach().numpy().tolist()[robot_index])
        RL_calf_pos_list.append(env.dof_pos[:, 2].cpu().detach().numpy().tolist()[robot_index])
        RL_wheel_pos_list.append(env.dof_pos[:, 3].cpu().detach().numpy().tolist()[robot_index])
        
        RL_hip_vel_list.append(env.dof_vel[:, 0].cpu().detach().numpy().tolist()[robot_index])
        RL_thigh_vel_list.append(env.dof_vel[:, 1].cpu().detach().numpy().tolist()[robot_index])
        RL_calf_vel_list.append(env.dof_vel[:, 2].cpu().detach().numpy().tolist()[robot_index])
        RL_wheel_vel_list.append(env.dof_vel[:, 3].cpu().detach().numpy().tolist()[robot_index])
        
        RL_hip_torque_list.append(env.torques[:, 0].cpu().detach().numpy().tolist()[robot_index])
        RL_thigh_torque_list.append(env.torques[:, 1].cpu().detach().numpy().tolist()[robot_index])
        RL_calf_torque_list.append(env.torques[:, 2].cpu().detach().numpy().tolist()[robot_index])
        RL_wheel_torque_list.append(env.torques[:, 3].cpu().detach().numpy().tolist()[robot_index])

        RR_hip_pos_list.append(env.dof_pos[:, 0].cpu().detach().numpy().tolist()[robot_index])    
        RR_thigh_pos_list.append(env.dof_pos[:, 1].cpu().detach().numpy().tolist()[robot_index])
        RR_calf_pos_list.append(env.dof_pos[:, 2].cpu().detach().numpy().tolist()[robot_index])
        RR_wheel_pos_list.append(env.dof_pos[:, 3].cpu().detach().numpy().tolist()[robot_index])
        
        RR_hip_vel_list.append(env.dof_vel[:, 0].cpu().detach().numpy().tolist()[robot_index])
        RR_thigh_vel_list.append(env.dof_vel[:, 1].cpu().detach().numpy().tolist()[robot_index])
        RR_calf_vel_list.append(env.dof_vel[:, 2].cpu().detach().numpy().tolist()[robot_index])
        RR_wheel_vel_list.append(env.dof_vel[:, 3].cpu().detach().numpy().tolist()[robot_index])
        
        RR_hip_torque_list.append(env.torques[:, 0].cpu().detach().numpy().tolist()[robot_index])
        RR_thigh_torque_list.append(env.torques[:, 1].cpu().detach().numpy().tolist()[robot_index])
        RR_calf_torque_list.append(env.torques[:, 2].cpu().detach().numpy().tolist()[robot_index])
        RR_wheel_torque_list.append(env.torques[:, 3].cpu().detach().numpy().tolist()[robot_index])

        FL_hip_pos_ref_list.append(env.ref_dof_pos[:, 0].cpu().detach().numpy().tolist()[robot_index])
        FL_thigh_pos_ref_list.append(env.ref_dof_pos[:, 1].cpu().detach().numpy().tolist()[robot_index])
        FL_calf_pos_ref_list.append(env.ref_dof_pos[:, 2].cpu().detach().numpy().tolist()[robot_index])
        FL_wheel_vel_ref_list.append(env.ref_dof_vel[:, 3].cpu().detach().numpy().tolist()[robot_index])

        if MOVE_CAMERA:
            camera_offset = np.array(env_cfg.viewer.pos)
            target_position = np.array(
                env.base_pos[robot_index, :].to(device="cpu")
            )
            camera_position = target_position + camera_offset
            env.set_camera(camera_position, target_position)
        if RENDER:
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
            img = np.reshape(img, (1080, 1920, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video.write(img[..., :3])
        
        motor_vel_list.append(env.dof_vel.cpu().detach().numpy().tolist()[0])
        ang_vel_list.append(env.base_ang_vel[robot_index, 2].item())
        feet_height_list.append((env.rigid_state[robot_index, 5, 2] - 0.14).item())
        air_time_list.append(env.feet_air_time[robot_index, 0].item())
        # feet_height_smooth_list.append(env.feet_height_smooth_sum[robot_index].item())
        for i in range(3):
            vel_list[i].append(env.base_lin_vel[robot_index, i].item())
        logger_pd.log_states(
            {
                "command_height": env.commands[robot_index, 2].item(),
                "base_height": env.base_height[robot_index].item(),
                "command_ang_vel": env.commands[robot_index, 1].item(),
                "base_ang_vel": env.base_ang_vel[robot_index, 2].item(),
                "command_vel_x": env.commands[robot_index, 0].item(),
                "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                "base_roll": env.base_euler_xyz[robot_index, 0].item(),
                "base_pitch": env.base_euler_xyz[robot_index, 1].item(),
                "base_yaw": env.base_euler_xyz[robot_index, 2].item(),
                #joint pos
                "FL_hip_pos": env.dof_pos[robot_index, 0].item(),
                "FL_thigh_pos": env.dof_pos[robot_index, 1].item(),
                "FL_calf_pos": env.dof_pos[robot_index, 2].item(),
                "FL_wheel_pos": env.dof_pos[robot_index, 3].item(),
                "FR_hip_pos": env.dof_pos[robot_index, 4].item(),
                "FR_thigh_pos": env.dof_pos[robot_index, 5].item(),
                "FR_calf_pos": env.dof_pos[robot_index, 6].item(),
                "FR_wheel_pos": env.dof_pos[robot_index, 7].item(),
                "RL_hip_pos": env.dof_pos[robot_index, 8].item(),
                "RL_thigh_pos": env.dof_pos[robot_index, 9].item(),
                "RL_calf_pos": env.dof_pos[robot_index, 10].item(),
                "RL_wheel_pos": env.dof_pos[robot_index, 11].item(),
                "RR_hip_pos": env.dof_pos[robot_index, 12].item(),
                "RR_thigh_pos": env.dof_pos[robot_index, 13].item(),
                "RR_calf_pos": env.dof_pos[robot_index, 14].item(),
                "RR_wheel_pos": env.dof_pos[robot_index, 15].item(),
                #joint vel
                "FL_hip_vel": env.dof_vel[robot_index, 0].item(),
                "FL_thigh_vel": env.dof_vel[robot_index, 1].item(),
                "FL_calf_vel": env.dof_vel[robot_index, 2].item(),
                "FL_wheel_vel": env.dof_vel[robot_index, 3].item(),
                "FR_hip_vel": env.dof_vel[robot_index, 4].item(),
                "FR_thigh_vel": env.dof_vel[robot_index, 5].item(),
                "FR_calf_vel": env.dof_vel[robot_index, 6].item(),
                "FR_wheel_vel": env.dof_vel[robot_index, 7].item(),
                "RL_hip_vel": env.dof_vel[robot_index, 8].item(),
                "RL_thigh_vel": env.dof_vel[robot_index, 9].item(),
                "RL_calf_vel": env.dof_vel[robot_index, 10].item(),
                "RL_wheel_vel": env.dof_vel[robot_index, 11].item(),
                "RR_hip_vel": env.dof_vel[robot_index, 12].item(),
                "RR_thigh_vel": env.dof_vel[robot_index, 13].item(),
                "RR_calf_vel": env.dof_vel[robot_index, 14].item(),
                "RR_wheel_vel": env.dof_vel[robot_index, 15].item(),
                # action
                "FL_hip_action": env.actions[robot_index, 0].item() * env_cfg.control.pos_action_scale,
                "FL_thigh_action": env.actions[robot_index, 1].item() * env_cfg.control.pos_action_scale,
                "FL_calf_action": env.actions[robot_index, 2].item() * env_cfg.control.pos_action_scale,
                "FL_wheel_action": env.actions[robot_index, 3].item() * env_cfg.control.vel_action_scale,
                "FR_hip_action": env.actions[robot_index, 4].item() * env_cfg.control.pos_action_scale,
                "FR_thigh_action": env.actions[robot_index, 5].item() * env_cfg.control.pos_action_scale,
                "FR_calf_action": env.actions[robot_index, 6].item() * env_cfg.control.pos_action_scale,
                "FR_wheel_action": env.actions[robot_index, 7].item() * env_cfg.control.vel_action_scale,
                "RL_hip_action": env.actions[robot_index, 8].item() * env_cfg.control.pos_action_scale,
                "RL_thigh_action": env.actions[robot_index, 9].item() * env_cfg.control.pos_action_scale,
                "RL_calf_action": env.actions[robot_index, 10].item() * env_cfg.control.pos_action_scale,
                "RL_wheel_action": env.actions[robot_index, 11].item() * env_cfg.control.vel_action_scale,
                "RR_hip_action": env.actions[robot_index, 12].item() * env_cfg.control.pos_action_scale,
                "RR_thigh_action": env.actions[robot_index, 13].item() * env_cfg.control.pos_action_scale,
                "RR_calf_action": env.actions[robot_index, 14].item() * env_cfg.control.pos_action_scale,
                "RR_wheel_action": env.actions[robot_index, 15].item() * env_cfg.control.vel_action_scale,
                #ref
                "FL_hip_ref_pos": env.ref_dof_pos[robot_index, 0].item(),
                "FL_thigh_ref_pos": env.ref_dof_pos[robot_index, 1].item(),
                "FL_calf_ref_pos": env.ref_dof_pos[robot_index, 2].item(),
                "FL_wheel_ref_vel": env.ref_dof_vel[robot_index, 3].item(),
                "FR_hip_ref_pos": env.ref_dof_pos[robot_index, 4].item(),
                "FR_thigh_ref_pos": env.ref_dof_pos[robot_index, 5].item(),
                "FR_calf_ref_pos": env.ref_dof_pos[robot_index, 6].item(),
                "FR_wheel_ref_vel": env.ref_dof_vel[robot_index, 7].item(),
                "RL_hip_ref_pos": env.ref_dof_pos[robot_index, 8].item(),
                "RL_thigh_ref_pos": env.ref_dof_pos[robot_index, 9].item(),
                "RL_calf_ref_pos": env.ref_dof_pos[robot_index, 10].item(),
                "RL_wheel_ref_vel": env.ref_dof_vel[robot_index, 11].item(),
                "RR_hip_ref_pos": env.ref_dof_pos[robot_index, 12].item(),
                "RR_thigh_ref_pos": env.ref_dof_pos[robot_index, 13].item(),
                "RR_calf_ref_pos": env.ref_dof_pos[robot_index, 14].item(),
                "RR_wheel_ref_vel": env.ref_dof_vel[robot_index, 15].item(),
                #torque
                "FL_hip_torque": env.torques[robot_index, 0].item(),
                "FL_thigh_torque": env.torques[robot_index, 1].item(),
                "FL_calf_torque": env.torques[robot_index, 2].item(),
                "FL_wheel_torque": env.torques[robot_index, 3].item(),
                "FR_hip_torque": env.torques[robot_index, 4].item(),
                "FR_thigh_torque": env.torques[robot_index, 5].item(),
                "FR_calf_torque": env.torques[robot_index, 6].item(),
                "FR_wheel_torque": env.torques[robot_index, 7].item(),
                "RL_hip_torque": env.torques[robot_index, 8].item(),
                "RL_thigh_torque": env.torques[robot_index, 9].item(),
                "RL_calf_torque": env.torques[robot_index, 10].item(),
                "RL_wheel_torque": env.torques[robot_index, 11].item(),
                "RR_hip_torque": env.torques[robot_index, 12].item(),
                "RR_thigh_torque": env.torques[robot_index, 13].item(),
                "RR_calf_torque": env.torques[robot_index, 14].item(),
                "RR_wheel_torque": env.torques[robot_index, 15].item(),    
            
            }
            )
        # ====================== Log states ======================
        if infos["episode"]:
            num_episodes = torch.sum(env.reset_buf).item()
            if num_episodes>0:
                logger_pd.log_rewards(infos["episode"], num_episodes)
    if RENDER:
        video.release()
    # print('size',len(actions_list))
    #FL_POS
    with open('./data/FL_hip_pos_list.csv','w') as file:
        json.dump(FL_hip_pos_list, file)
    with open('./data/FL_thigh_pos_list.csv','w') as file:
        json.dump(FL_thigh_pos_list, file)
    with open('./data/FL_calf_pos_list.csv','w') as file:
        json.dump(FL_calf_pos_list, file)
    with open('./data/FL_wheel_pos_list.csv','w') as file:
        json.dump(FL_wheel_pos_list, file)
    #FL_VEL
    with open('./data/FL_hip_vel_list.csv','w') as file:
        json.dump(FL_hip_vel_list, file)
    with open('./data/FL_thigh_vel_list.csv','w') as file:
        json.dump(FL_thigh_vel_list, file)
    with open('./data/FL_calf_vel_list.csv','w') as file:
        json.dump(FL_calf_vel_list, file)
    with open('./data/FL_wheel_vel_list.csv','w') as file:
        json.dump(FL_wheel_vel_list, file)
    #FL_TORQUE
    with open('./data/FL_hip_torque_list.csv','w') as file:
        json.dump(FL_hip_torque_list, file)
    with open('./data/FL_thigh_torque_list.csv','w') as file:
        json.dump(FL_thigh_torque_list, file)
    with open('./data/FL_calf_torque_list.csv','w') as file:
        json.dump(FL_calf_torque_list, file)
    with open('./data/FL_wheel_torque_list.csv','w') as file:
        json.dump(FL_wheel_torque_list, file)
    #FR_POS
    with open('./data/FR_hip_pos_list.csv','w') as file:
        json.dump(FR_hip_pos_list, file)
    with open('./data/FR_thigh_pos_list.csv','w') as file:
        json.dump(FR_thigh_pos_list, file)
    with open('./data/FR_calf_pos_list.csv','w') as file:
        json.dump(FR_calf_pos_list, file)
    with open('./data/FR_wheel_pos_list.csv','w') as file:
        json.dump(FR_wheel_pos_list, file)
    #FR_VEL
    with open('./data/FR_hip_vel_list.csv','w') as file:
        json.dump(FR_hip_vel_list, file)
    with open('./data/FR_thigh_vel_list.csv','w') as file:
        json.dump(FR_thigh_vel_list, file)
    with open('./data/FR_calf_vel_list.csv','w') as file:
        json.dump(FR_calf_vel_list, file)
    with open('./data/FR_wheel_vel_list.csv','w') as file:
        json.dump(FR_wheel_vel_list, file)
    #FR_TORQUE
    with open('./data/FR_hip_torque_list.csv','w') as file:
        json.dump(FR_hip_torque_list, file)
    with open('./data/FR_thigh_torque_list.csv','w') as file:
        json.dump(FR_thigh_torque_list, file)
    with open('./data/FR_calf_torque_list.csv','w') as file:
        json.dump(FR_calf_torque_list, file)
    with open('./data/FR_wheel_torque_list.csv','w') as file:
        json.dump(FR_wheel_torque_list, file)
    #RL_POS
    with open('./data/RL_hip_pos_list.csv','w') as file:
        json.dump(RL_hip_pos_list, file)
    with open('./data/RL_thigh_pos_list.csv','w') as file:
        json.dump(RL_thigh_pos_list, file)
    with open('./data/RL_calf_pos_list.csv','w') as file:
        json.dump(RL_calf_pos_list, file)
    with open('./data/RL_wheel_pos_list.csv','w') as file:
        json.dump(RL_wheel_pos_list, file)
    #RL_VEL
    with open('./data/RL_hip_vel_list.csv','w') as file:
        json.dump(RL_hip_vel_list, file)
    with open('./data/RL_thigh_vel_list.csv','w') as file:
        json.dump(RL_thigh_vel_list, file)
    with open('./data/RL_calf_vel_list.csv','w') as file:
        json.dump(RL_calf_vel_list, file)
    with open('./data/RL_wheel_vel_list.csv','w') as file:
        json.dump(RL_wheel_vel_list, file)
    #RL_TORQUE
    with open('./data/RL_hip_torque_list.csv','w') as file:
        json.dump(RL_hip_torque_list, file)
    with open('./data/RL_thigh_torque_list.csv','w') as file:
        json.dump(RL_thigh_torque_list, file)
    with open('./data/RL_calf_torque_list.csv','w') as file:
        json.dump(RL_calf_torque_list, file)
    with open('./data/RL_wheel_torque_list.csv','w') as file:
        json.dump(RL_wheel_torque_list, file)
    #RR_POS
    with open('./data/RR_hip_pos_list.csv','w') as file:
        json.dump(RR_hip_pos_list, file)
    with open('./data/RR_thigh_pos_list.csv','w') as file:
        json.dump(RR_thigh_pos_list, file)
    with open('./data/RR_calf_pos_list.csv','w') as file:
        json.dump(RR_calf_pos_list, file)
    with open('./data/RR_wheel_pos_list.csv','w') as file:
        json.dump(RR_wheel_pos_list, file)
    #RR_VEL
    with open('./data/RR_hip_vel_list.csv','w') as file:
        json.dump(RR_hip_vel_list, file)
    with open('./data/RR_thigh_vel_list.csv','w') as file:
        json.dump(RR_thigh_vel_list, file)
    with open('./data/RR_calf_vel_list.csv','w') as file:
        json.dump(RR_calf_vel_list, file)
    with open('./data/RR_wheel_vel_list.csv','w') as file:
        json.dump(RR_wheel_vel_list, file)
    #RR_TORQUE
    with open('./data/RR_hip_torque_list.csv','w') as file:
        json.dump(RR_hip_torque_list, file)
    with open('./data/RR_thigh_torque_list.csv','w') as file:
        json.dump(RR_thigh_torque_list, file)
    with open('./data/RR_calf_torque_list.csv','w') as file:
        json.dump(RR_calf_torque_list, file)
    with open('./data/RR_wheel_torque_list.csv','w') as file:
        json.dump(RR_wheel_torque_list, file)
    #ref
    with open('./data/FL_hip_pos_ref_list.csv','w') as file:
        json.dump(FL_hip_pos_ref_list, file)
    with open('./data/FL_thigh_pos_ref_list.csv','w') as file:
        json.dump(FL_thigh_pos_ref_list, file)
    with open('./data/FL_calf_pos_ref_list.csv','w') as file:
        json.dump(FL_calf_pos_ref_list, file)
    with open('./data/FL_wheel_vel_ref_list.csv','w') as file:
        json.dump(FL_wheel_vel_ref_list, file)

    print("--------------------------------")
    print("--------------------------------")
    print("被clip的次数: ", count) 
    logger_pd.print_rewards()
    
    
if __name__ == '__main__':
    x_vel_cmd , y_vel_cmd , height_cmd = 0., 0., 0.
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
   
    EXPORT_POLICY = False
    RENDER = True
    FIX_COMMAND = False
    MOVE_CAMERA = False
    args = get_args()
    args.task="nezha2_pd"
    args.num_envs = 1
    args.experiment_name = 'nezha2_pd'   
    args.load_run="test"
    # args.checkpoints = 4000
    play(args)