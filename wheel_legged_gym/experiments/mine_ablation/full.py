"""full ablation configuration."""
from .common import CommonCfg, CommonTrainCfg


class FullCfg(CommonCfg):
    pass


class FullTrainCfg(CommonTrainCfg):
    class runner(CommonTrainCfg.runner):
        run_name = 'full'
