import torch.nn as nn
import torch
import numpy as np
from .utils import get_activation, check_cnnoutput
from torch.distributions import Normal
from torch.nn import functional

class estimator(nn.Module):
    def __init__(
        self,
        num_obs,
        output_dim=3,
        activation="elu",
        estimator_hidden_dims=[512, 256, 256],
        device = "cuda",
        target_frames=1,
        critic_frame_size=None
    ):
        super().__init__()
        self.device = device
        self.num_obs = num_obs
        self.target_frames = target_frames
        self.critic_frame_size = critic_frame_size

        self.mlp_est = MLP(
            num_obs,
            output_dim,
            activation,
            estimator_hidden_dims,
        )


    def forward(self, obs_history):
        output= self.mlp_est(obs_history)
        return output

    def targets(self, critic_obs, num_est_prob):
        if self.target_frames == 1:
            return critic_obs[:, -num_est_prob:]
        if (not self.critic_frame_size
                or critic_obs.shape[-1] % self.critic_frame_size
                or critic_obs.shape[-1] // self.critic_frame_size < self.target_frames):
            raise ValueError("Critic history does not contain the required target frames")
        frames = critic_obs.reshape(critic_obs.shape[0], -1, self.critic_frame_size)
        return frames[:, -self.target_frames:, -num_est_prob:].reshape(critic_obs.shape[0], -1)

    def loss_fn(self, estimator_latent, critic_obs, num_est_prob):

        prob_real = self.targets(critic_obs, num_est_prob)
        # print("prob_real",prob_real.shape)
        vel_loss = functional.mse_loss(estimator_latent, prob_real, reduction="none").mean(-1)
        Difference = estimator_latent.cpu().detach().numpy() - prob_real.cpu().detach().numpy()
        v_avg_diff_x ,v_avg_diff_y, v_avg_diff_z, base_height_diff= np.mean(np.abs(Difference).reshape(-1, num_est_prob),axis=0)
        # print(critic_obs.shape)
        loss = vel_loss
        return {
            "loss": loss,
            "v_avg_diff_x": v_avg_diff_x,
            "v_avg_diff_y": v_avg_diff_y,
            "v_avg_diff_z": v_avg_diff_z,
            "base_height_diff": base_height_diff,
        }


class MLP(nn.Module):
    def __init__(self, input_size, output_size, activation, hidden_dims):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        module = []
        module.append(nn.Linear(self.input_size, hidden_dims[0]))
        module.append(self.activation)
        for i in range(len(hidden_dims) - 1):
            module.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            module.append(self.activation)
        module.append(nn.Linear(hidden_dims[-1], self.output_size))
        self.encoder = nn.Sequential(*module)

    def forward(self, obs_history):
        RS_obs_history = obs_history.reshape(obs_history.shape[0],-1)
        return self.encoder(RS_obs_history)

