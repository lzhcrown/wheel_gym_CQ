import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as torchd
from torch.distributions import Normal, Categorical


class MINEEstimator(nn.Module):
    def __init__(self,
                 temporal_steps,
                 num_one_step_obs,
                 num_one_step_privileged_obs,
                 enc_hidden_dims=[256, 128, 64, 16], # 原[128, 64, 16]
                 tar_hidden_dims=[256, 128, 64], # 原[128, 64]
                 num_est_prob = 3,
                 is_privileged_obs=False,
                 activation='elu',
                 learning_rate=1e-3,
                 max_grad_norm=10.0,
                 num_prototype=32,#32
                 temperature=3.0,#3.0
                 use_mode_gate=True,
                 mode_loss_coef=0.5,
                 **kwargs):
        if kwargs:
            print("Estimator_CL.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(MINEEstimator, self).__init__()
        activation = get_activation(activation)

        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.num_one_step_privileged_obs = num_one_step_privileged_obs
        self.history_obs_dim = self.temporal_steps * self.num_one_step_obs
        self.num_latent = enc_hidden_dims[-1]
        self.max_grad_norm = max_grad_norm
        self.num_est_prob = num_est_prob
        self.is_privileged_obs = is_privileged_obs
        self.temperature = temperature
        self.use_mode_gate = use_mode_gate
        self.mode_loss_coef = float(mode_loss_coef)
        if not math.isfinite(self.mode_loss_coef) or self.mode_loss_coef < 0:
            raise ValueError("mode_loss_coef must be finite and nonnegative")
        if not self.use_mode_gate and self.mode_loss_coef != 0:
            raise ValueError("Disabling the mode gate also requires mode_loss_coef=0")

        # Encoder
        enc_input_dim = self.history_obs_dim
        enc_layers = []
        for l in range(len(enc_hidden_dims) - 1):
            enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[l]), activation]
            enc_input_dim = enc_hidden_dims[l]
        enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[-1] + self.num_est_prob)]
        self.encoder = nn.Sequential(*enc_layers)

        # Instantiate every branch in every ablation to keep initialization
        # identical. An ablated branch is bypassed and receives no gradients.
        # Source and gate use the same history, ordered oldest to newest.
        self.mode_gate = nn.Sequential(
            nn.Linear(self.history_obs_dim, 64),
            nn.ELU(),
            nn.Linear(64, 3) # 输出 3 个值，对应三个大类
        )
                # --- 新增：Mode Prototypes ---
        # 定义 3 个可学习的模态原型向量，维度与隐变量 z 一致
        # 这些原型将代表三个大类的“中心”特征
        self.mode_prototypes = nn.Parameter(torch.randn(3, enc_hidden_dims[-1]))
        # --------------------------------

        # Target
        if self.is_privileged_obs:
            tar_input_dim = self.num_one_step_privileged_obs
        else:
            tar_input_dim = self.num_one_step_obs
        tar_layers = []
        for l in range(len(tar_hidden_dims)):
            tar_layers += [nn.Linear(tar_input_dim, tar_hidden_dims[l]), activation]
            tar_input_dim = tar_hidden_dims[l]
        tar_layers += [nn.Linear(tar_input_dim, enc_hidden_dims[-1])]
        self.target = nn.Sequential(*tar_layers)

        # Prototype
        self.proto = nn.Embedding(num_prototype, enc_hidden_dims[-1])

        # Optimizer
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_latent(self, obs_history):
        vel, z = self.encode(obs_history)
        return vel.detach(), z.detach()

    def forward(self, obs_history):
        vel, z = self.encode(obs_history)
        return vel.detach(), z.detach()

    def encode(self, obs_history):
        vel, z, _, _ = self._encode_history(obs_history)
        return vel, z

    def _encode_history(self, obs_history):
        """Share the representation between auxiliary learning and policy use.

        Normalization is differentiable and leaves the prototype parameters
        untouched. Only forward() detaches the outputs consumed by the policy.
        """
        history = obs_history.detach()
        parts = self.encoder(history)
        vel, z = parts[..., :self.num_est_prob], parts[..., self.num_est_prob:]

        if not self.use_mode_gate:
            return vel, F.normalize(z, dim=-1, p=2), None, None

        mode_logits = self.mode_gate(history)
        mode_probs = F.softmax(mode_logits, dim=-1)
        normalized_prototypes = F.normalize(self.mode_prototypes, dim=-1, p=2)
        weighted_proto = mode_probs @ normalized_prototypes
        z = F.normalize(z * weighted_proto, dim=-1, p=2)
        return vel, z, mode_probs, normalized_prototypes

    def update(self, obs_history, critic_obs, next_critic_obs, lr=None):
        """Align history through t with the current privileged frame p_t.

        State regression keeps its existing next-state velocity/height target.
        The target encoder uses only the latest frame in the pre-action Critic
        history, including when next_critic_obs contains a terminal snapshot.
        """
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
                
        vel = next_critic_obs[:, -self.num_est_prob:].detach()

        current_privileged_frame = critic_obs.detach()[:, -self.num_one_step_privileged_obs:]
        if self.is_privileged_obs:
            target_obs = current_privileged_frame
        else:
            target_obs = current_privileged_frame[:, :self.num_one_step_obs]
 
        pred_vel, z_s, mode_probs, normalized_prototypes = self._encode_history(obs_history)
        z_t = self.target(target_obs)
        z_t = F.normalize(z_t, dim=-1, p=2)

        with torch.no_grad():
            w = self.proto.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.proto.weight.copy_(w)
            # --- 修改点 2：移除对 mode_prototypes 的原地更新 ---
            # 我们不再在这里更新 self.mode_prototypes，而是每次使用时通过 F.normalize 获取归一化版本
            # 这样避免了在计算图中修改需要梯度的参数
            # --------------------------------

        score_s = z_s @ self.proto.weight.T
        score_t = z_t @ self.proto.weight.T

        with torch.no_grad():
            q_s = sinkhorn(score_s)
            q_t = sinkhorn(score_t)

        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)

        swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()
        estimation_loss = F.mse_loss(pred_vel, vel)
        
        losses = estimation_loss + swap_loss
        if self.mode_loss_coef > 0:
            similarity = torch.matmul(z_s.detach(), normalized_prototypes.T)
            mode_loss = F.cross_entropy(similarity, torch.argmax(mode_probs, dim=-1))
            losses = losses + self.mode_loss_coef * mode_loss

        self.optimizer.zero_grad()
        losses.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return estimation_loss.item(), swap_loss.item()


@torch.no_grad()
def sinkhorn(out, eps=0.05, iters=3):
    Q = torch.exp(out / eps).T
    K, B = Q.shape[0], Q.shape[1]
    Q /= Q.sum()

    for it in range(iters):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    return (Q * B).T


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "silu":
        return nn.SiLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
