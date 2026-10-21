
from wheel_legged_gym.envs.nezha2.nezha2_config import Nezha2Cfg,Nezha2CfgPPO

class Nezha2Cfg_VAE(Nezha2Cfg):
    class env(Nezha2Cfg.env):
        frame_stack = 17#VAE输入的历史观测帧数
        c_frame_stack = 3#critic输入的历史观测帧数
        # vae_stack = 6 #VAE输入的历史观测帧数
        num_latent = 16
        num_est_prob = 3
        num_est_v_h = 4#vel_xyz, height预测的信息的总维度，加入到privileged_obs中
        num_single_obs = 65#3+3+3+12+16+12+16
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = num_single_obs +16+16+1+3+16+16+51+187+4##3+3+3+12+16+12+16+16+16+1+3+16+16+51+187+4
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
    # class terrain_cfg_VAE(Nezha2Cfg.terrain):
    #     mesh_type = 'plane'

class Nezha2CfgPPO_VAE(Nezha2CfgPPO):
    seed = 10
    runner_class_name = 'OnPolicyRunnerVAE'
    

    class policy(Nezha2CfgPPO.policy):
        # VAE para
        # encoder_hidden_dims=[128,64]
        encoder_hidden_dims=[32,16]
        #目前最优：encoder_hidden_dims=[128,64]
        #目前最优：decoder_hidden_dims=[32,64]
        # decoder_hidden_dims=[64,128]
        decoder_hidden_dims=[16,32]
    class algorithm(Nezha2CfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-3#1e-5
        num_learning_epochs = 5 ##2
        gamma = 0.99 #0.994
        lam = 0.95#0.9
        num_mini_batches = 4

        # VAE para
        vae_learning_rate = 1.e-3#5.e-4
        kl_weight = 1
        value_loss_coef = 1.0
        max_grad_norm_vae = 0.7#7e-1
        num_adaptation_module_substeps = 1

    class runner:
        policy_class_name = 'ActorCritic_VAE'
        algorithm_class_name = 'PPO_VAE'
        num_steps_per_env = 60  # per iteration
        max_iterations = 100001  # number of policy updates        #  xxw

        # logging
        save_interval = 500  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'nezha2_vae'
        run_name = 'vae'
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = '/home/luntui_mine'  # updated from load_run and chkpt