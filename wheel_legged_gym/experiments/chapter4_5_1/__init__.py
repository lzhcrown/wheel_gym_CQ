"""Six tasks for thesis section 4.5.1."""
from .baseline import BaselineCfg, BaselineTrainCfg
from .critic_truth import CriticTruthCfg, CriticTruthTrainCfg
from .actor_critic_truth import ActorCriticTruthCfg, ActorCriticTruthTrainCfg
from .est_128_64 import Est12864Cfg, Est12864TrainCfg
from .est_64_64 import Est6464Cfg, Est6464TrainCfg
from .est_64_32 import Est6432Cfg, Est6432TrainCfg

CONFIGS = {
    "exp451_baseline": (BaselineCfg, BaselineTrainCfg),
    "exp451_critic_truth": (CriticTruthCfg, CriticTruthTrainCfg),
    "exp451_actor_critic_truth": (ActorCriticTruthCfg, ActorCriticTruthTrainCfg),
    "exp451_est_128_64": (Est12864Cfg, Est12864TrainCfg),
    "exp451_est_64_64": (Est6464Cfg, Est6464TrainCfg),
    "exp451_est_64_32": (Est6432Cfg, Est6432TrainCfg),
}


def register_tasks(registry):
    from .env import Chapter451Env
    for task, (env_cfg, train_cfg) in CONFIGS.items():
        registry.register(task, Chapter451Env, env_cfg(), train_cfg())
