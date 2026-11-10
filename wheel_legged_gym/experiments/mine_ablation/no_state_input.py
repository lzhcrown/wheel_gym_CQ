"""no_state_input ablation configuration."""
from .common import CommonCfg, CommonTrainCfg


class NoStateInputCfg(CommonCfg):
    pass


class NoStateInputTrainCfg(CommonTrainCfg):
    class policy(CommonTrainCfg.policy):
        use_state_input = False

    class runner(CommonTrainCfg.runner):
        run_name = 'no_state_input'
