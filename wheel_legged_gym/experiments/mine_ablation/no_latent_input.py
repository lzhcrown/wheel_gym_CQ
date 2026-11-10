"""no_latent_input ablation configuration."""
from .common import CommonCfg, CommonTrainCfg


class NoLatentInputCfg(CommonCfg):
    pass


class NoLatentInputTrainCfg(CommonTrainCfg):
    class policy(CommonTrainCfg.policy):
        use_latent_input = False

    class runner(CommonTrainCfg.runner):
        run_name = 'no_latent_input'
