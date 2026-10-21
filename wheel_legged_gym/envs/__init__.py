# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR, WHEEL_LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot

import os

from wheel_legged_gym.utils.task_registry import task_registry


# <><><><><><>nezha
from .nezha2.nezha2_config import Nezha2Cfg, Nezha2CfgPPO
from .nezha2.nezha2_env import Nezha2FreeEnv
task_registry.register(
    "nezha2", Nezha2FreeEnv, Nezha2Cfg(), Nezha2CfgPPO()
)
#<><><><><><>nezha_pd
from .nezha2_pd.nezha2_pd_config import Nezha2Cfg_PD, Nezha2CfgPPO_PD
from .nezha2_pd.nezha2_pd_env import Nezha2FreeEnv_PD
task_registry.register(
    "nezha2_pd", Nezha2FreeEnv_PD, Nezha2Cfg_PD(), Nezha2CfgPPO_PD()
)
#<><><><><><>nezha_est
from .nezha2_est.nezha2_est_config import Nezha2Cfg_EST, Nezha2CfgPPO_EST
from .nezha2.nezha2_env import Nezha2FreeEnv
task_registry.register(
    "nezha2_est", Nezha2FreeEnv, Nezha2Cfg_EST(), Nezha2CfgPPO_EST()
)
#<><><><><><>nezha_vae
from .nezha2_vae.nezha2_vae_config import Nezha2Cfg_VAE, Nezha2CfgPPO_VAE
from .nezha2.nezha2_env import Nezha2FreeEnv
task_registry.register(
    "nezha2_vae", Nezha2FreeEnv, Nezha2Cfg_VAE(), Nezha2CfgPPO_VAE()
)
#<><><><><><>nezha_vae_github
from .nezha2_vae_github.nezha2_vae_github_config import Nezha2Cfg_VAE_github, Nezha2CfgPPO_VAE_github
from .nezha2_vae_github.nezha2_env import Nezha2FreeEnv
task_registry.register(
    "nezha2_vae_github", Nezha2FreeEnv, Nezha2Cfg_VAE_github(), Nezha2CfgPPO_VAE_github()
)
#<><><><><><>nezha_rma
from .nezha2_rma.nezha2_rma_config import Nezha2Cfg_RMA, Nezha2CfgPPO_RMA
from .nezha2.nezha2_env import Nezha2FreeEnv
task_registry.register(
    "nezha2_rma", Nezha2FreeEnv, Nezha2Cfg_RMA(), Nezha2CfgPPO_RMA()
)
#<><><><><><>nezha_est_force
from .nezha2_est_force.nezha2_est_force_config import Nezha2Cfg_EST_FORCE, Nezha2CfgPPO_EST_FORCE
from .nezha2.nezha2_env import Nezha2FreeEnv
task_registry.register(
    "nezha2_est_force", Nezha2FreeEnv, Nezha2Cfg_EST_FORCE(), Nezha2CfgPPO_EST_FORCE()
)
#<><><><><><>nezha_TS_teacher
from .nezha2_ts_teacher.nezha2_ts_teacher_config import Nezha2Cfg_t, Nezha2CfgPPO_t
from .nezha2_ts_teacher.nezha2_env import Nezha2FreeEnv
task_registry.register(
    "nezha2_ts_teacher", Nezha2FreeEnv, Nezha2Cfg_t(), Nezha2CfgPPO_t()
)
#<><><><><><>nezha_TS_student
from .nezha2_ts_student.nezha2_ts_student_config import Nezha2Cfg_s, Nezha2CfgPPO_s
from .nezha2_ts_student.nezha2_env import Nezha2FreeEnv
task_registry.register(
    "nezha2_ts_student", Nezha2FreeEnv, Nezha2Cfg_s(), Nezha2CfgPPO_s()
)
#<><><><><><>nezha_mine
from .nezha2_mine.nezha2_mine_config import Nezha2Cfg_MINE, Nezha2CfgPPO_MINE
from .nezha2.nezha2_env import Nezha2FreeEnv
task_registry.register(
    "nezha2_mine", Nezha2FreeEnv, Nezha2Cfg_MINE(), Nezha2CfgPPO_MINE()
)

# Experiment suites have separate namespaces and share the current robot.
from wheel_legged_gym.experiments.chapter4_5_1 import register_tasks as register_451_tasks
register_451_tasks(task_registry)
