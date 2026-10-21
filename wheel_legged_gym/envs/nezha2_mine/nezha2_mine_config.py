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


# from wheel_legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from wheel_legged_gym.envs.nezha2.nezha2_config import Nezha2Cfg,Nezha2CfgPPO


class Nezha2Cfg_MINE(Nezha2Cfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(Nezha2Cfg.env):
        # change the observation dim
        num_est_v_h = 4
        num_est_force = None
        num_est_prob = None
        num_est_prob = 4 # 显示估计的信息的维度，请把需要估计的信息放在特权观测的最后
                        # 注意： MINE的特权观测在compute_privileged_observations函数和compute_observations函数中均有使用，请保证两者观测信息顺序相同

        frame_stack = 2
        c_frame_stack = 3
        num_single_obs = 65 #3+3+3+12+16+12+16
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = num_single_obs +16+16+1+3+16+16+51+187+4##3+3+3+12+16+12+16+16+16+1+3+16+16+51+187+4
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        episode_length_s = 20  # episode length in seconds
        use_ref_actions = False
       
class Nezha2CfgPPO_MINE(Nezha2CfgPPO):
    seed = 10
    runner_class_name = 'MINEOnPolicyRunner'
    

    class policy:
        init_noise_std = 1.0
        # actor_hidden_dims = [256, 128, 64]
        # critic_hidden_dims = [256, 128, 64]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        enc_hidden_dims = [256, 128, 64, 16] # 最后一层为latent维度
        tar_hidden_dims = [256, 128, 64]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # max_grad_norm = 1.
        max_grad_norm = 10

    class algorithm(Nezha2CfgPPO.algorithm):
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        

    class runner:
        policy_class_name = 'MINEActorCritic'
        algorithm_class_name = 'MINEPPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 100000 # number of policy updates

        # logging
        save_interval = 200 # check for potential saves every this many iterations
        experiment_name = 'nezha2_est' # log保存路径的标识
        run_name = 'rough-1'
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkp