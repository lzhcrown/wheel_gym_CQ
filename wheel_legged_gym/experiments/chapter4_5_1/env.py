"""One environment implementation for all six comparisons."""
import torch
from wheel_legged_gym.envs.nezha2.nezha2_env import Nezha2FreeEnv


class Chapter451Env(Nezha2FreeEnv):
    def _augment_actor_observations(self, obs):
        if self.cfg.env.actor_true_state:
            return torch.cat((obs,
                              self.base_lin_vel * self.obs_scales.lin_vel,
                              self.base_height_obs * self.obs_scales.height_measurements), dim=-1)
        return obs
