"""Load only the algorithm requested by the task."""
from importlib import import_module

_EXPORTS = {'VecEnv': '.vec_env',
 'PPO': '.ppo_ts.ppo',
 'OnPolicyRunner': '.ppo.on_policy_runner',
 'ActorCritic': '.ppo_ts.actor_critic',
 'RolloutStorage': '.ppo_ts.rollout_storage',
 'PPO_Estimator': '.state_estimate.ppo_est',
 'OnPolicyRunnerEstimator': '.state_estimate.on_policy_runner_est',
 'ActorCritic_Estimator': '.state_estimate.actor_critic_est',
 'RolloutStorage_Estimator': '.state_estimate.rollout_storage_est',
 'PPO_Estimator_Force': '.state_estimate_force.ppo_est',
 'OnPolicyRunnerEstimator_Force': '.state_estimate_force.on_policy_runner_est',
 'ActorCritic_Estimator_Force': '.state_estimate_force.actor_critic_est',
 'RolloutStorage_Estimator_Force': '.state_estimate_force.rollout_storage_est',
 'PPO_VAE': '.DreamWaQ.ppo_vae',
 'OnPolicyRunnerVAE': '.DreamWaQ.on_policy_runner_vae',
 'ActorCritic_VAE': '.DreamWaQ.actor_critic_vae',
 'RolloutStorage_VAE': '.DreamWaQ.rollout_storage_vae',
 'OnPolicyRunner_VAE_github': '.DreamWaq_github.on_policy_runner',
 'ActorCritic_DWAQ': '.DreamWaq_github.actor_critic_DWAQ',
 'PPO_RMA': '.RMA.ppo_rma',
 'OnPolicyRunnerRMA': '.RMA.on_policy_runner_rma',
 'ActorCritic_RMA': '.RMA.actor_critic_rma',
 'RolloutStorage_RMA': '.RMA.rollout_storage_rma',
 'MINEPPO': '.Mine.ppo_mine',
 'MINEOnPolicyRunner': '.Mine.on_policy_runner_mine',
 'MINEActorCritic': '.Mine.actor_critic_mine',
 'MINERolloutStorage': '.Mine.rollout_storage_mine',
 'OnPolicyRunner_ts': '.ppo_ts.on_policy_runner_ts'}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
