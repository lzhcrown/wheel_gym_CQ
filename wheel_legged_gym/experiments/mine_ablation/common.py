"""Shared settings; ablations change only declared policy switches."""
from wheel_legged_gym.envs.nezha2_mine.nezha2_mine_config import (
    Nezha2Cfg_MINE, Nezha2CfgPPO_MINE,
)


class CommonCfg(Nezha2Cfg_MINE):
    pass


class CommonTrainCfg(Nezha2CfgPPO_MINE):
    class policy(Nezha2CfgPPO_MINE.policy):
        use_latent_input = True
        use_state_input = True
        use_mode_gate = True
        mode_loss_coef = 0.5

    class runner(Nezha2CfgPPO_MINE.runner):
        experiment_name = "mine_ablation"
        csv_tags = ["Train/mean_reward"]
