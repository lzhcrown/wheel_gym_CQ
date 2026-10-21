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
                 **kwargs):
        if kwargs:
            print("Estimator_CL.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(MINEEstimator, self).__init__()
        activation = get_activation(activation)

        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.num_one_step_privileged_obs = num_one_step_privileged_obs
        self.num_latent = enc_hidden_dims[-1]
        self.max_grad_norm = max_grad_norm
        self.num_est_prob = num_est_prob
        self.is_privileged_obs = is_privileged_obs
        self.temperature = temperature

        # Encoder
        enc_input_dim = self.temporal_steps * self.num_one_step_obs
        enc_layers = []
        for l in range(len(enc_hidden_dims) - 1):
            enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[l]), activation]
            enc_input_dim = enc_hidden_dims[l]
        enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[-1] + self.num_est_prob)]
        self.encoder = nn.Sequential(*enc_layers)

        # --- 新增：Mode Gating Network ---
        # 输入为当前时刻的单步观测，输出为 [0, 1] 之间的门控权重
        self.mode_gate = nn.Sequential(
            nn.Linear(self.num_one_step_obs, 64),
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
        parts = self.encoder(obs_history.detach())
        vel, z = parts[..., :self.num_est_prob], parts[..., self.num_est_prob:]
        
        # --- 修改：门控逻辑 ---
        current_obs = obs_history.detach()[:, 0:self.num_one_step_obs]
        mode_logits = self.mode_gate(current_obs)
        mode_probs = F.softmax(mode_logits, dim=-1)
        weighted_proto = torch.matmul(mode_probs, self.mode_prototypes)
        z = z * weighted_proto
        # ------------------------

        z = F.normalize(z, dim=-1, p=2)
        return vel.detach(), z.detach()

    def encode(self, obs_history):
        parts = self.encoder(obs_history.detach())
        vel, z = parts[..., :self.num_est_prob], parts[..., self.num_est_prob:]
        
        # --- 修改：门控逻辑 ---
        current_obs = obs_history.detach()[:, 0:self.num_one_step_obs]
        mode_logits = self.mode_gate(current_obs)
        mode_probs = F.softmax(mode_logits, dim=-1)
        weighted_proto = torch.matmul(mode_probs, self.mode_prototypes)
        z = z * weighted_proto
        # ------------------------
        
        z = F.normalize(z, dim=-1, p=2)
        return vel, z

    def update(self, obs_history, next_critic_obs, lr=None):
        if lr is not None:
            self.learning_rate = lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate
                
        vel = next_critic_obs[:, -self.num_est_prob:].detach()  # vel信息存在特权信息中的倒数第四到倒数第二维

        if self.is_privileged_obs:
            next_obs = next_critic_obs.detach()[:, -self.num_one_step_privileged_obs:]
        else:
            next_obs = next_critic_obs.detach()[:, -self.num_one_step_privileged_obs:-self.num_one_step_privileged_obs+self.num_one_step_obs]
 
        z_s = self.encoder(obs_history)
        z_t = self.target(next_obs)
        pred_vel, z_s = z_s[..., :self.num_est_prob], z_s[..., self.num_est_prob:]

        #当前观测
        current_obs = obs_history.detach()[:, 0:self.num_one_step_obs]
        # 1. 获取门控 Logits 并计算 Softmax 概率
        mode_logits = self.mode_gate(current_obs) # [Batch, 3]
        mode_probs = F.softmax(mode_logits, dim=-1) # [Batch, 3]
        
        # --- 修改点 1：使用函数式归一化，避免原地修改参数 ---
        # 获取归一化后的模态原型，但不修改 self.mode_prototypes 本身
        # F.normalize 返回一个新的张量，不会修改输入
        normalized_prototypes = F.normalize(self.mode_prototypes, dim=-1, p=2)
        
        # 2. 计算加权模态原型
        # 将隐变量 z 乘以对应类别的原型权重
        # normalized_prototypes: [3, Latent_Dim]
        # mode_probs: [Batch, 3]
        # weighted_proto: [Batch, Latent_Dim]
        weighted_proto = torch.matmul(mode_probs, normalized_prototypes)
        
        # 3. 应用门控
        z_s = z_s * weighted_proto
        # ------------------------

        z_s = F.normalize(z_s, dim=-1, p=2)
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
        
        # --- 修改点 3：计算模态分类损失时也使用归一化后的原型 ---
        # 计算隐变量 z_s 与各个模态原型的相似度
        # z_s: [Batch, Latent_Dim], normalized_prototypes: [3, Latent_Dim]
        # similarity: [Batch, 3]
        similarity = torch.matmul(z_s.detach(), normalized_prototypes.T)
        
        # 目标是让 z_s 与其被分配到的原型（mode_probs 指向的那个）尽可能相似
        mode_loss = F.cross_entropy(similarity, torch.argmax(mode_probs, dim=-1))
        # ------------------------

        losses = estimation_loss + swap_loss + mode_loss * 0.5 # 添加模态损失，权重可调

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