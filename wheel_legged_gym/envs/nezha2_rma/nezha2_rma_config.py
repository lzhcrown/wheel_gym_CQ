
from wheel_legged_gym.envs.nezha2.nezha2_config import Nezha2Cfg,Nezha2CfgPPO


class Nezha2Cfg_RMA(Nezha2Cfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(Nezha2Cfg.env):
        # rma para
        # xxx frame_stack用于控制存储的历史大小，用于输入给adaptation encoder
        frame_stack = 15
        # xxx 这个actor_input_stack用于控制输入给actor的观测数量，应该小于frame_stack(总的历史)
        actor_input_stack = 1

        # xxx c_frame_stack用于控制存储的历史大小,用于输入给critic
        c_frame_stack = 3
        num_latent = 12
        num_est_v_h = 4#vel_xyz, height预测的信息的总维度，加入到privileged_obs中
        # 用于控制输入给expert的维度（高度+xyz+height）
        num_expert_input = 187 + 4
        
        num_single_obs = 65 #3+3+3+12+16+12+16 
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = num_single_obs +16+16+1+3+16+16+51+187+4
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
    # class terrain_cfg_VAE(Nezha2Cfg.terrain):
    #     mesh_type = 'plane'



class Nezha2CfgPPO_RMA(Nezha2CfgPPO):
    seed = 10
    runner_class_name = 'OnPolicyRunnerRMA'
    

    class policy(Nezha2CfgPPO.policy):
        # rma para
        expert_hidden_dims=[256, 128]
        adaptation_hidden_dims=[256, 128]

    class algorithm(Nezha2CfgPPO.algorithm):
        # rma para
        adaptation_module_learning_rate=5e-4
        num_adaptation_module_substeps = 3
        
    class runner:
        policy_class_name = 'ActorCritic_RMA'
        algorithm_class_name = 'PPO_RMA'
        num_steps_per_env = 60  # per iteration
        max_iterations = 1000001  # number of policy updates        #  xxw

        # logging
        save_interval = 50  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'nezha2_rma'
        run_name = 'rma'
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = '/home/luntui_mine'  # updated from load_run and chkpt