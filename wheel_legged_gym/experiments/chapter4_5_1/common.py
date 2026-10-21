"""Shared settings for all six explicit-state estimation comparisons.

Rewards, controller, terrain, PPO hyperparameters and rollout budget inherit
from the current Nezha configuration. Only information access and estimator
width are varied by this suite.
"""
from wheel_legged_gym.envs.nezha2.nezha2_config import Nezha2Cfg, Nezha2CfgPPO
from wheel_legged_gym.envs.nezha2_est.nezha2_est_config import Nezha2CfgPPO_EST


class CommonCfg(Nezha2Cfg):
    class env(Nezha2Cfg.env):
        actor_true_state = False


class CommonTrainCfg(Nezha2CfgPPO):
    class runner(Nezha2CfgPPO.runner):
        experiment_name = "chapter4_5_1"


class CriticTruthCfg(CommonCfg):
    class env(CommonCfg.env):
        num_est_v_h = 4
        num_est_prob = 4
        single_num_privileged_obs = CommonCfg.env.single_num_privileged_obs + 4
        num_privileged_obs = CommonCfg.env.c_frame_stack * single_num_privileged_obs


class EstimatorCfg(CriticTruthCfg):
    class env(CriticTruthCfg.env):
        actor_input_stack = CommonCfg.env.frame_stack


class EstimatorTrainCfg(CommonTrainCfg):
    runner_class_name = "OnPolicyRunnerEstimator"

    class policy(CommonTrainCfg.policy):
        estimated_frames = CommonCfg.env.frame_stack
        critic_frame_size = CriticTruthCfg.env.single_num_privileged_obs

    class algorithm(CommonTrainCfg.algorithm):
        mlp_learning_rate = Nezha2CfgPPO_EST.algorithm.mlp_learning_rate
        num_adaptation_module_substeps = Nezha2CfgPPO_EST.algorithm.num_adaptation_module_substeps

    class runner(CommonTrainCfg.runner):
        policy_class_name = "ActorCritic_Estimator"
        algorithm_class_name = "PPO_Estimator"
