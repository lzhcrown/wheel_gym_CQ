# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

"""Replay the gated dual-encoder policy; optionally export its full inference path."""
import argparse
import copy
from pathlib import Path
import sys

import isaacgym  # Isaac Gym must be imported before torch.
import torch

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR, envs  # Register before utils imports.
from wheel_legged_gym.algo.Mine.actor_critic_mine import build_actor_input


class MineInferencePolicy(torch.nn.Module):
    """Use the same estimator and Actor operations as MINEActorCritic.act_inference.

    The estimator includes its mode gate and learned mode prototypes. Reusing its
    forward method also preserves the original observation order and scaling.
    """
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor).cpu()
        self.estimator = copy.deepcopy(actor_critic.estimator).cpu()
        self.num_single_obs = actor_critic.num_one_step_obs
        self.num_observations = actor_critic.num_actor_obs
        self.use_state_input = actor_critic.use_state_input
        self.use_latent_input = actor_critic.use_latent_input
        self.eval()

    def forward(self, obs_history):
        velocity_height, latent = self.estimator(obs_history)
        actor_input = build_actor_input(
            obs_history, velocity_height, latent, self.num_single_obs,
            self.use_state_input, self.use_latent_input)
        return self.actor(actor_input)

    def export(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        example = torch.zeros(1, self.num_observations, dtype=next(self.actor.parameters()).dtype)
        with torch.inference_mode():
            policy = torch.jit.trace(self, example)
        destination = directory / "policy_mine.pt"
        policy.save(str(destination))
        return destination


def get_play_args(argv=None):
    """Add replay options while preserving the shared simulator/checkpoint CLI."""
    from wheel_legged_gym.utils.helpers import get_args
    from wheel_legged_gym.utils import task_registry

    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False, description=__doc__)
    parser.add_argument("--play-steps", type=int, default=1000,
                        help="Number of simulation steps to replay (default: 1000)")
    parser.add_argument("--log-root", type=Path,
                        help="Checkpoint root containing run directories; defaults to logs/EXPERIMENT")
    parser.add_argument("--export-policy", action="store_true",
                        help="Export the complete gated policy as TorchScript")
    parser.add_argument("--export-dir", type=Path,
                        help="Export directory (default: selected run/exported)")
    options, remaining = parser.parse_known_args(argv)
    if options.play_steps < 0:
        parser.error("--play-steps must be nonnegative")
    if any(option in remaining for option in ("-h", "--help")):
        print(parser.format_help())
    previous_argv = sys.argv
    try:
        sys.argv = [previous_argv[0], *remaining]
        args = get_args(defaults={"task": "nezha2_mine", "num_envs": 1})
    finally:
        sys.argv = previous_argv
    if args.task not in task_registry.task_classes:
        parser.error("Unknown training task: " + args.task)
    _, train_cfg = task_registry.get_cfgs(args.task)
    if train_cfg.runner_class_name != "MINEOnPolicyRunner":
        parser.error("Replay requires a MINE training task")
    if args.num_envs <= 0:
        parser.error("--num_envs must be positive")
    for name, value in vars(options).items():
        setattr(args, name, value)
    return args


def resolve_checkpoint(args, train_cfg):
    """Resolve the user's run/checkpoint before creating the simulation."""
    from wheel_legged_gym.utils.helpers import get_load_path

    root = args.log_root
    if root is None:
        root = Path(WHEEL_LEGGED_GYM_ROOT_DIR) / "logs" / train_cfg.runner.experiment_name
    return Path(get_load_path(
        root, load_run=train_cfg.runner.load_run,
        checkpoint=train_cfg.runner.checkpoint, task=args.task,
    ))


def play(args):
    from wheel_legged_gym.utils import task_registry
    from wheel_legged_gym.utils.helpers import update_cfg_from_args

    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    if train_cfg.runner_class_name != "MINEOnPolicyRunner":
        raise ValueError("Replay requires a MINE training task")
    env_cfg, train_cfg = update_cfg_from_args(env_cfg, train_cfg, args)
    checkpoint = resolve_checkpoint(args, train_cfg)
    print("Loading policy from: {}".format(checkpoint))

    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    try:
        # Replay creates no training run and loads only model parameters.
        runner_args = copy.copy(args)
        runner_args.resume = False
        train_cfg.runner.resume = False
        runner, _ = task_registry.make_alg_runner(
            env=env, name=args.task, args=runner_args,
            train_cfg=train_cfg, log_root=None,
        )
        runner.load(str(checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=env.device)
        if args.export_policy:
            export_dir = args.export_dir if args.export_dir is not None else checkpoint.parent / "exported"
            destination = MineInferencePolicy(runner.alg.actor_critic).export(export_dir)
            print("Exported policy to: {}".format(destination))
        if not args.headless:
            env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)
        obs = env.get_observations()
        with torch.inference_mode():
            for _ in range(args.play_steps):
                actions = policy(obs)
                obs, _, _, _, _ = env.step(actions)
    finally:
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    play(get_play_args())
