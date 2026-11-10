"""no_gate ablation configuration."""
from .common import CommonCfg, CommonTrainCfg


class NoGateCfg(CommonCfg):
    pass


class NoGateTrainCfg(CommonTrainCfg):
    class policy(CommonTrainCfg.policy):
        use_mode_gate = False
        mode_loss_coef = 0.0

    class runner(CommonTrainCfg.runner):
        run_name = 'no_gate'
