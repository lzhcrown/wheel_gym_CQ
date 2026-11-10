"""Five controlled ablations of the gated modal dual encoder."""
from .full import FullCfg, FullTrainCfg
from .no_latent_input import NoLatentInputCfg, NoLatentInputTrainCfg
from .no_gate import NoGateCfg, NoGateTrainCfg
from .no_state_input import NoStateInputCfg, NoStateInputTrainCfg
from .no_mode_loss import NoModeLossCfg, NoModeLossTrainCfg

CONFIGS = {
    "mine_ablation_full": (FullCfg, FullTrainCfg),
    "mine_ablation_no_latent_input": (NoLatentInputCfg, NoLatentInputTrainCfg),
    "mine_ablation_no_gate": (NoGateCfg, NoGateTrainCfg),
    "mine_ablation_no_state_input": (NoStateInputCfg, NoStateInputTrainCfg),
    "mine_ablation_no_mode_loss": (NoModeLossCfg, NoModeLossTrainCfg),
}


def register_tasks(registry):
    from wheel_legged_gym.envs.nezha2.nezha2_env import Nezha2FreeEnv
    for task, (env_cfg_class, train_cfg_class) in CONFIGS.items():
        registry.register(task, Nezha2FreeEnv, env_cfg_class(), train_cfg_class())
