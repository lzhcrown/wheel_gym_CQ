"""no_mode_loss ablation configuration."""
from .common import CommonCfg, CommonTrainCfg


class NoModeLossCfg(CommonCfg):
    pass


class NoModeLossTrainCfg(CommonTrainCfg):
    class policy(CommonTrainCfg.policy):
        mode_loss_coef = 0.0

    class runner(CommonTrainCfg.runner):
        run_name = 'no_mode_loss'
