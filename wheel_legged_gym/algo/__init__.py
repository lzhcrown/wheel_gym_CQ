"""Load only the algorithm requested by the task."""
from importlib import import_module

_EXPORTS = {
    'VecEnv': '.vec_env',
    'MINEPPO': '.Mine.ppo_mine',
    'MINEOnPolicyRunner': '.Mine.on_policy_runner_mine',
    'MINEActorCritic': '.Mine.actor_critic_mine',
    'MINERolloutStorage': '.Mine.rollout_storage_mine',
}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
