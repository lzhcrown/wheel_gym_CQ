from .common import CriticTruthCfg, CommonTrainCfg


class ActorCriticTruthCfg(CriticTruthCfg):
    class env(CriticTruthCfg.env):
        actor_true_state = True
        num_single_obs = CriticTruthCfg.env.num_single_obs + 4
        num_observations = CriticTruthCfg.env.frame_stack * num_single_obs
        # Critic already contains velocity/height; do not add them twice.


class ActorCriticTruthTrainCfg(CommonTrainCfg):
    class runner(CommonTrainCfg.runner):
        run_name = "actor_critic_truth"
