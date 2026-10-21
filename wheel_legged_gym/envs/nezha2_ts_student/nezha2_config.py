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


from .legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Nezha2Cfg(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        num_envs = 4096 # 强化学习同时训练智能体的数量
        num_actions = 16 # 可操控的动作数量
        # change the observation dim
        num_est_v_h = None
        num_est_force = None
        num_est_prob = None
        frame_stack = 2
        c_frame_stack = 3
        num_single_obs = 65 #3+3+3+12+16+12+16
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = num_single_obs +16+16+1+3+16+16+51+187+4##3+3+3+12+16+12+16+16+16+1+3+16+16+51+187+4
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        episode_length_s = 20  # episode length in seconds
        use_ref_actions = False
        fail_to_terminal_time_s = 0.6
    # 机器人指令类
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True  
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.5
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 10.
        terrain_width = 10.
        num_rows = 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [plane; slope; uniform, stair_up,stair_down, discrete]
        # terrain_proportions = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]
        # terrain_proportions = [0.0, 0.0, 0.5, 0.5, 0.0]
        terrain_proportions = [0.1, 0.1, 0.35, 0.2, 0.25]
        slope_treshold = 0.75 #slopes above this threshold will be corrected to vertical surfaces
        track_test = False # 测试柏林噪声是否添加成功
        add_perlin_noise = False # 开启时需要将机器人初始位姿提高，避免机器人初始姿态陷入地面

    class safety:
        # safety factors
        pos_limit = 1
        vel_limit = 1
        torque_limit = 1    #0.85 xxx 1
        # acc_limit = 0.6
        # dof_acc_limits_ratio = 6

        # TODO
    class noise:
        add_noise = True
        noise_level = 1    # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.6]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 0.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        rand_init_dof = False
        rand_init_dof_range = 0.
        default_joint_angles = {  # target angles when action = 0.0
            'FL_hip_joint': -0.1,#-0.1,   # [rad]
            'RL_hip_joint': -0.1,#0.1,   # [rad]
            'FR_hip_joint': 0.1,#-0.1,  # [rad]
            'RR_hip_joint': 0.1,#0.1,   # [rad]

            'FL_thigh_joint': 0.9250,     # [rad]
            'RL_thigh_joint': 0.9250,   # [rad]
            'FR_thigh_joint': 0.9250,     # [rad]
            'RR_thigh_joint': 0.9250,   # [rad]

            'FL_calf_joint': -1.8500,   # [rad]
            'RL_calf_joint': -1.8500,    # [rad]
            'FR_calf_joint': -1.8500,  # [rad]
            'RR_calf_joint': -1.8500,    # [rad]
            
            'FL_foot_joint':0.0,
            'RL_foot_joint':0.0,
            'FR_foot_joint':0.0,
            'RR_foot_joint':0.0,
        }
        # init_joint_angles = { # = target angles [rad] when action = 0.0
        #     'FL_hip_joint': 0.0,   # [rad]
        #     'RL_hip_joint': 0.0,   # [rad]
        #     'FR_hip_joint': 0.0 ,  # [rad]
        #     'RR_hip_joint': 0.0,   # [rad]

        #     'FL_thigh_joint': 0.9250,     # [rad]
        #     'RL_thigh_joint': -0.9250,   # [rad]
        #     'FR_thigh_joint': 0.9250,     # [rad]
        #     'RR_thigh_joint': -0.9250,   # [rad]

        #     'FL_calf_joint': -1.8500,   # [rad]
        #     'RL_calf_joint': 1.8500,    # [rad]
        #     'FR_calf_joint': -1.8500,  # [rad]
        #     'RR_calf_joint': 1.8500,    # [rad]
            
        #     'FL_foot_joint':0.0,
        #     'RL_foot_joint':0.0,
        #     'FR_foot_joint':0.0,
        #     'RR_foot_joint':0.0,
        # }


    # 机器人关节电机控制模式、参数
    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P' # 位置控制、速度控制、扭矩控制
        
        stiffness = {'hip_joint': 150.,'thigh_joint': 150.,'calf_joint': 300.,"foot_joint":0}  # [N*m/rad] 刚度系数k_p 
        damping = {'hip_joint': 4,'thigh_joint': 4,'calf_joint': 4,"foot_joint":1.2}     # [N*m*s/rad] 阻尼系数k_d
        # action scale: target angle = actionScale * action + defaultAngle
        # 乘一个缩放因子，目的是让动作值适应不同关节的运动范围
        # action_scale = 0.25
        vel_action_scale = 10.0 # 轮子的速度缩放超参数
        pos_action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        # 仿真环境的控制频率/decimation = 实际环境中的控制频率       
        decimation = 4  

        # ratio: self.action = ratio * self.action + (1 - ratio) * last_actoin
        action_smoothness = False
        ratio = 0.9
        projected_gravity = False
        # feedforward_force = 6 # 42.46kg / 2
    # 与机器人urdf相关参数
    class asset(LeggedRobotCfg.asset):   
        file = "{WHEEL_LEGGED_GYM_ROOT_DIR}/resources/robots/urdf2.1/urdf/nezha3_description.urdf"     
        name = "nezha2"
        offset = 0.
        l1 = 0.0
        l2 = 0.0
        foot_name = "foot"
        wheel_name = ["foot"]
        penalize_contacts_on = ["hip","thigh", "calf", "thunk"] # 惩罚接触
        terminate_after_contacts_on = ["thunk","hip"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter "base","calf","hip","thigh"
        replace_cylinder_with_capsule = False
        flip_visual_attachments = False #

        disable_gravity = False
        fix_base_link = False
        fix_base_link_height = 1.8  # fix the base of the robot at the height


    class sim(LeggedRobotCfg.sim):
        web_vis = False
        port = 6001      # zmp的端口，如果训练启动卡住，则这个端口被占用，更改以下就行
        web_vis_envs = 1 #用于控制web可视化的机器人个数，建议不要超过5个，性能影响很大
        keep_default_viewer = False

        dt = 0.005  # 200 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10                                  
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand(LeggedRobotCfg.domain_rand):

        rand_interval_s = 4    ## 
        # TODO randomize_default_dof_pos, randomize_action_delay
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 0.5  # 0.2
        max_push_ang_vel = 0.5
        #刚体摩擦力
        randomize_friction = True            
        friction_range = [0.2, 1.25]
        #随机化base_link的质量
        randomize_base_mass = True #
        # randomize_mass_range = [0.5, 1.5]         # 乘负载
        added_mass_range = [-5, 5]              # 加负载
        #随机化刚体的质心，True表示随机化base_link的质心
        randomize_com_displacement = True      #加到priv里的东西
        com_displacement_range = [-0.05, 0.05]  # base link com的随机化范围

        action_noise = 0.0 # 0.02
        action_delay = 0. # 0.1
        
        #随机化间隔时间
        #是否随机化刚体的friction和restitution
        randomize_rigids_after_start = False     #控制link的friction和restituion的随机化开关       

        #刚体恢复系数
        randomize_restitution = False            
        restitution_range = [0, 1.0]            
        

        #是否随机每一个link的质心，True表示每个link的质心
        randomize_each_link = False
        link_com_displacement_range_factor = 0.02   # link com的随机化比例(与com_displacement_range相乘)
        
        #随机化每个link的质量和inertia
        randomize_inertia = False    
        randomize_inertia_range = [0.8, 1.2]

        #输出的torque乘的随机化系数
        randomize_motor_strength = False      
        motor_strength_range = [0.9, 1.1]      

        #action转torque的Kp，Kd系数随机化
        randomize_PD_factor = False #             
        Kp_factor_range = [0.9, 1.1]            
        Kd_factor_range = [0.9, 1.1]

        #目前是使用torque的offset
        randomize_motor_offset = False 

        default_motor_offset = [0,0,0,0,\
                                0,0,0,0,\
                                0,0,0,0,\
                                0,0,0,0]
        motor_offset_range = [-0.03, 0.03]

        # defautl dof pos位置没变，但数值上有rand的偏差
        randomize_default_dof_pos = False 
        randomize_default_dof_pos_range = [-0.1, 0.1]

        # 随机化重力
        gravity_rand_interval_s = 7
        gravity_impulse_duration = 1.0
        randomize_gravity = False # 建议不加
        gravity_range = [-1.0, 1.0]         #

        #下面重新定义了，可删除
        randomize_lag_timesteps = False      # 模拟delay，对于lag用于给历史的action
        lag_timesteps = 2       #2~4ms walk these ways 加固定action延迟
        
        #torque的delay
        randomize_torque_delay = False
        torque_delay_steps = 2

        # randomize_obs_delay = False #用队列加固定obs延迟
        # obs_delay_steps = 1

        # True表示添加action的延迟
        add_lag = False
        # True表示reset时随机选择一个延迟步数，并且保持不变
        randomize_lag_timesteps = False
        # True表示每个step都随机选择一个延迟步数
        randomize_lag_timesteps_perstep = False
        lag_timesteps_range = [3, 7]

        # True表示添加dof的延迟
        add_dof_lag = False
        # True表示reset时随机选择一个延迟步数，并且保持不变
        randomize_dof_lag_timesteps = False
        # True表示每个step都随机选择一个延迟步数
        randomize_dof_lag_timesteps_perstep = False
        dof_lag_timesteps_range = [0, 2] # 1~4ms

        # True表示添加imu的延迟
        add_imu_lag = False # 现在是euler，需要projected gravity
        # True表示reset时随机选择一个延迟步数，并且保持不变
        randomize_imu_lag_timesteps = False
        # True表示每个step都随机选择一个延迟步数
        randomize_imu_lag_timesteps_perstep = False# 不常用always False
        imu_lag_timesteps_range = [5, 11] # 实际10~22ms

        #True表示随机化关节摩擦力
        randomize_coulomb_friction = False
        joint_stick_friction_range = [0.1, 0.2]
        joint_coulomb_friction_range = [0.0, 0.0]
        
        #关节friction
        randomize_joint_friction = True
        randomize_joint_friction_each_joint = False
                                
        # default_joint_friction = [0.0, 0.0, 0.0, 0.0, \
        #                           0.0, 0.0, 0.0, 0.0, \
        #                           0.0, 0.0, 0.0, 0.0, \
        #                           0.0, 0.0, 0.0, 0.0 ]
        default_joint_friction = [0.025, 0.05, 0.05, 0.05, \
                                  0.025, 0.05, 0.05, 0.05, \
                                  0.025, 0.05, 0.05, 0.05, \
                                  0.025, 0.05, 0.05, 0.05 ]
        # # self.dof names: ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FL_foot_joint', 
        #                  'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'FR_foot_joint',
        #                  'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RL_foot_joint', 
        #                  'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', 'RR_foot_joint']
        joint_friction_range = [0.95, 1.05]
        # joint_friction_range = [1.5, 1.5]
        # joint_1_friction_range = [0.9, 1.1]
        # joint_2_friction_range = [0.9, 1.1]
        # joint_3_friction_range = [0.9, 1.1]
        # joint_4_friction_range = [0.9, 1.1]
        # joint_5_friction_range = [0.9, 1.1]
        # joint_6_friction_range = [0.9, 1.1]

        randomize_joint_damping = False
        randomize_joint_damping_each_joint = False
        default_joint_damping = [0.0, 0.0, 0.0, 0.0, \
                                 0.0, 0.0, 0.0, 0.0, \
                                 0.0, 0.0, 0.0, 0.0, \
                                 0.0, 0.0, 0.0, 0.0 ]
        # default_joint_damping = [5.0, 10.0, 10, 0.0, \
        #                          5.0, 10.0, 10, 0.0, \
        #                          5.0, 10.0, 10, 0.0, \
        #                          5.0, 10.0, 10, 0.0 ]
        # default_joint_damping = [0, 0.0, 10, 0.0, \
        #                          0, 0.0, 10, 0.0, \
        #                          0, 0.0, 10, 0.0, \
        #                          0, 0.0, 10, 0.0 ]
        # # self.dof names: ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FL_foot_joint', 
        #                  'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'FR_foot_joint',
        #                  'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RL_foot_joint', 
        #                  'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', 'RR_foot_joint']
        joint_damping_range = [0.95, 1.05]
        # joint_1_damping_range = [0.8, 1.2]
        # joint_2_damping_range = [0.8, 1.2] 
        # joint_3_damping_range = [0.8, 1.2]
        # joint_4_damping_range = [0.8, 1.2]
        # joint_5_damping_range = [0.8, 1.2]
        # joint_6_damping_range = [0.8, 1.2]

        #关节aramture
        randomize_joint_armature = False 
        randomize_joint_armature_each_joint = False
        default_joint_armature = [0.0, 0.0, 0.0, 0.0, \
                                  0.0, 0.0, 0.0, 0.0, \
                                  0.0, 0.0, 0.0, 0.0, \
                                  0.0, 0.0, 0.0, 0.0 ]
        # default_joint_armature = [1.0, 2.0, 1.0, 0.0, \
        #                           1.0, 2.0, 1.0, 0.0, \
        #                           1.0, 2.0, 1.0, 0.0, \
        #                           1.0, 2.0, 1.0, 0.0 ]
        # default_joint_armature = [0.0, 0.0, 1.0, 0.0, \
        #                           0.0, 0.0, 1.0, 0.0, \
        #                           0.0, 0.0, 1.0, 0.0, \
        #                           0.0, 0.0, 1.0, 0.0 ]
        # # self.dof names: ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', 'FL_foot_joint', 
        #                  'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'FR_foot_joint',
        #                  'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint', 'RL_foot_joint', 
        #                  'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', 'RR_foot_joint']
        joint_armature_range = [0.95, 1.05]     # Factor
        # joint_1_armature_range = [0.95, 1.05]
        # joint_2_armature_range = [0.95, 1.05]
        # joint_3_armature_range = [0.95, 1.05]
        # joint_4_armature_range = [0.95, 1.05]
        # joint_5_armature_range = [0.9, 1.1]
        # joint_6_armature_range = [0.9, 1.1]
        

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 1.5
        max_curriculum_yaw = 1.5 # 旋转指令课程的上限
        num_commands = 4
        resampling_time = 10.  # time before command are changed[s]
        # stand_still = False # increse the proportion of command "0"
        # stand_still_ratio = 0.8
        heading_command = False  # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s] x轴方向线速度
            lin_vel_y = [-0.6, 0.6]   # min max [m/s] y轴方向线速度
            ang_vel_yaw = [-1.0, 1.0]    # min max [r ad/s] 角速度
            heading = [-3.14, 3.14] # 航向 实际上没有使用这个维度

    class rewards:
        only_positive_rewards = False
        # tracking_sigma = 4    # vel = 0.5 对应 20; vel = 1 对应 4; vel = 1.5 对应 2; vel = 2 对应 1; vel = 3 对应 0.5
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 1.
        base_height_target = 0.5
        max_contact_force = 400. # forces above this value are penalized
        clip_single_reward = 10

        class scales:
            termination = -0.8 #  
            tracking_lin_vel = 3.0 # 速度跟踪
            tracking_ang_vel = 1.5 # 角速度跟踪
            lin_vel_z = -4.0 # 惩罚机器人在Z轴上的速度 对应现象为机器人上下起伏很大
            ang_vel_xy = -0.2 # 惩罚机器人在X轴和Y轴上的角速度 对应现象为遏制机器人左右晃动和前后晃动
            orientation = -2 # 强烈鼓励机器人与初始姿态的基座方向一致
            torques = -2.5e-6 # 机器人运控各电机输出的力矩的平方和 让模型找到最省力矩的方案
            dof_vel = -2e-3#-2e-3
            dof_acc = -2.5e-7
            base_height = -1.0 # 惩罚基座高度不保持在期望的高度上
            feet_air_time = 1.0 # ls:1.0;bt:4.0
            feet_contact_uniform = -0.1
            collision = -0.1
            feet_stumble = -0.5
            stand_still = -3.0
            dof_pos_limits = -0.9
            # hip_action_l2 = -0.9
            hip_default = -8 # 惩罚髋关节不在默认位置
            run_still = -0.1 #ls:-0.5;bt-0.1
            action_rate = -0.01#-0.01
            smoothness=-0.05
            # diagonal_wheel_sync = 0.1
            joint_power=-2e-4
            feet_contact_forces = -1.0e-3
            wheel_contact_vel_y = -1

    class normalization:
        contact_force_range = [0.0, 200.0]
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            dof_acc = 1.0
            height_measurements = 5.0
            torques = 1.0
            quat = 1.0
            force_measurements = 0.01

        clip_observations = 100.
        clip_actions = 100.
        


class Nezha2CfgPPO(LeggedRobotCfgPPO):
    seed = 10
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

    class policy():
        init_noise_std = 1.0
        # actor_hidden_dims = [256, 128, 256]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        # critic_hidden_dims = [512, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        value_loss_coef = 1.0#价值损失系数，代表价值估计的误差对总损失的影响程度1.0
        use_clipped_value_loss = True#是否使用裁剪的价值损失True
        clip_param = 0.2# 裁剪参数，用于裁剪价值估计的误差0.2
        entropy_coef = 0.01#熵损失系数，代表策略探索程度0.01
        num_learning_epochs = 5#学习轮数5
        num_mini_batches = 4  # 小批量数量mini batch size = num_envs*nsteps / num_mini_batches
        learning_rate = 1e-3  # 学习率5.e-4
        schedule = "adaptive"  # 学习率调度调整方式could be adaptive, fixed
        gamma = 0.99#折扣因子0.99
        lam = 0.95#广义优势估计参数0.95
        desired_kl = 0.01#目标KL散度0.01
        max_grad_norm = 1.0#最大梯度范围1.0

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # per iteration
        max_iterations = 1000001  # number of policy updates        #  xxw

        # logging
        save_interval = 100  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'nezha2'
        run_name = 'ppo'
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = '/home/luntui_mine'  # updated from load_run and chkpt
