
from wheel_legged_gym.envs.nezha2.nezha2_config import Nezha2Cfg, Nezha2CfgPPO


class Nezha2Cfg_EST_FORCE(Nezha2Cfg):
    class env(Nezha2Cfg.env):
        frame_stack = 5         #输入给encoder的
        actor_input_stack = 2  #输入给actor的
        num_est_v_h = 4        #vel_xyz, height预测的信息的总维度
        num_est_force = 8      #外部作用力的维度
        num_est_prob =num_est_v_h + num_est_force
        c_frame_stack = 3
        num_single_obs = 65 #3+3+3+12+16+12+16 
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = num_single_obs +num_est_prob + 16 + 16 + 1 + 3 + 16 + 16 + 51 + 187
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)

class Nezha2CfgPPO_EST_FORCE(Nezha2CfgPPO):
    runner_class_name = 'OnPolicyRunnerEstimator_Force'
    

    class policy(Nezha2CfgPPO.policy):
        init_noise_std = 1.0
        # estimator para
        estimator_hidden_dims=[64, 64]

    class algorithm(Nezha2CfgPPO.algorithm):
        # estimator para
        mlp_learning_rate = 1.e-3
        num_adaptation_module_substeps = 1
        
    class runner:
        policy_class_name = 'ActorCritic_Estimator_Force'
        algorithm_class_name = 'PPO_Estimator_Force'
        num_steps_per_env = 24  # per iteration
        max_iterations = 1000001  # number of policy updates        #  xxw

        # logging
        save_interval = 100  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'nezha2_test'
        run_name = 'test'
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = '/home/luntui_mine'  # updated from load_run and chkpt