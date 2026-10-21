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

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value


class Logger_pd_no_ref:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        #一次性注册所有key
        # self.state_log ={
        #     "FL_wheel_vel":[],
            
        # }
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if "rew" in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()

  


    def _plot(self):
        # nb_rows = 4
        # nb_cols = 4
        fig, axs = plt.subplots(3, 4, figsize=(10, 7))  # 图零 hip torque/vel
        fig, axs1 = plt.subplots(3, 4, figsize=(10, 7)) # 图一 thigh torque/vel
        fig, axs2 = plt.subplots(3, 4, figsize=(10, 7)) # 图二 calf torque/vel
        fig, axs3 = plt.subplots(3, 4, figsize=(10, 7)) # 图三 wheel torque/vel
        fig, axs4 = plt.subplots(2, 2, figsize=(10, 7)) # 图四 FL跟踪
        fig, axs5 = plt.subplots(2, 2, figsize=(10, 7)) # 图五 FR跟踪
        fig, axs6 = plt.subplots(2, 2, figsize=(10, 7)) # 图六 RL跟踪
        fig, axs7 = plt.subplots(2, 2, figsize=(10, 7)) # 图七 RR跟踪
        fig, axs8 = plt.subplots(2, 2, figsize=(10, 7)) # 图八 IMU角度
        for key, value in self.state_log.items():
            time = np.linspace(0, len(value) * self.dt, len(value))
            break
        log = self.state_log
        ######图零 hip torque/vel#######
        
        # 力矩 时间
        a = axs[0, 0]
        if log["FL_hip_torque"] != []:
            a.plot(time, log["FL_hip_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="FL_Hip_Torque [Nm]", title="FL_Hip_Torque")
        a.legend()

        # 速度 时间
        a = axs[1, 0]
        if log["FL_hip_vel"] != []:
            a.plot(time, log["FL_hip_vel"], label="FL_hip_vel")
        a.set(xlabel="time [s]", ylabel="FL_Hip_Vel[rad/s]", title="FL_Hip_Velocity")
        a.legend()

        # 力矩 速度
        a = axs[2, 0]
        if log["FL_hip_vel"] != [] and log["FL_hip_torque"] != []:
            a.plot(log["FL_hip_vel"], log["FL_hip_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="FL Torque/velocity curves")
        a.legend()

        # 力矩 时间
        a = axs[0, 1]
        if log["FR_hip_torque"] != []:
            a.plot(time, log["FR_hip_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="FR_Hip_Torque [Nm]", title="FR_Hip_Torque")
        a.legend()

        # 速度 时间
        a = axs[1, 1]
        if log["FR_hip_vel"] != []:
            a.plot(time, log["FR_hip_vel"], label="FR_hip_vel")
        a.set(xlabel="time [s]", ylabel="FR_Hip_Vel[rad/s]", title="FR_Hip_Velocity")
        a.legend()

        # 力矩 速度
        a = axs[2, 1]
        if log["FR_hip_vel"] != [] and log["FR_hip_torque"] != []:
            a.plot(log["FR_hip_vel"], log["FR_hip_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="FR Torque/velocity curves")
        a.legend()

        # 力矩 时间
        a = axs[0, 2]
        if log["RL_hip_torque"] != []:
            a.plot(time, log["RL_hip_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="RL_Hip_Torque [Nm]", title="RL_Hip_Torque")
        a.legend()

        # 速度 时间
        a = axs[1, 2]
        if log["RL_hip_vel"] != []:
            a.plot(time, log["RL_hip_vel"], label="RL_hip_vel")
        a.set(xlabel="time [s]", ylabel="RL_Hip_Vel[rad/s]", title="RL_Hip_Velocity")
        a.legend()

        # 力矩 速度
        a = axs[2, 2]
        if log["RL_hip_vel"] != [] and log["RL_hip_torque"] != []:
            a.plot(log["RL_hip_vel"], log["RL_hip_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="RL Torque/velocity curves")
        a.legend()

        # 力矩 时间
        a = axs[0, 3]
        if log["RR_hip_torque"] != []:
            a.plot(time, log["RR_hip_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="RR_Hip_Torque [Nm]", title="RR_Hip_Torque")
        a.legend()

        # 速度 时间
        a = axs[1, 3]
        if log["RR_hip_vel"] != []:
            a.plot(time, log["RR_hip_vel"], label="RR_hip_vel")
        a.set(xlabel="time [s]", ylabel="RR_Hip_Vel[rad/s]", title="RR_Hip_Velocity")
        a.legend()

        # 力矩 速度
        a = axs[2, 3]
        if log["RR_hip_vel"] != [] and log["RR_hip_torque"] != []:
            a.plot(log["RR_hip_vel"], log["RR_hip_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="RR Torque/velocity curves")
        a.legend()
        plt.tight_layout()
        
        ######图一 thigh torque/vel#######
        
        # 力矩 时间
        a = axs1[0, 0]
        if log["FL_thigh_torque"] != []:
            a.plot(time, log["FL_thigh_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="FL_Thigh_Torque [Nm]", title="FL_Thigh_Torque")
        a.legend()

        # 速度 时间
        a = axs1[1, 0]
        if log["FL_thigh_vel"] != []:
            a.plot(time, log["FL_thigh_vel"], label="FL_thigh_vel")
        a.set(xlabel="time [s]", ylabel="FL_Thigh_Vel[rad/s]", title="FL_Thigh_Velocity")
        a.legend()

        # 力矩 速度
        a = axs1[2, 0]
        if log["FL_thigh_vel"] != [] and log["FL_thigh_torque"] != []:
            a.plot(log["FL_thigh_vel"], log["FL_thigh_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="FL Torque/velocity curves")
        a.legend()

        # 力矩 时间
        a = axs1[0, 1]
        if log["FR_thigh_torque"] != []:
            a.plot(time, log["FR_thigh_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="FR_Thigh_Torque [Nm]", title="FR_Thigh_Torque")
        a.legend()

        # 速度 时间
        a = axs1[1, 1]
        if log["FR_thigh_vel"] != []:
            a.plot(time, log["FR_thigh_vel"], label="FR_thigh_vel")
        a.set(xlabel="time [s]", ylabel="FR_Thigh_Vel[rad/s]", title="FR_Thigh_Velocity")
        a.legend()

        # 力矩 速度
        a = axs1[2, 1]
        if log["FR_thigh_vel"] != [] and log["FR_thigh_torque"] != []:
            a.plot(log["FR_thigh_vel"], log["FR_thigh_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="FR Torque/velocity curves")
        a.legend()

        # 力矩 时间
        a = axs1[0, 2]
        if log["RL_thigh_torque"] != []:
            a.plot(time, log["RL_thigh_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="RL_Thigh_Torque [Nm]", title="RL_Thigh_Torque")
        a.legend()

        # 速度 时间
        a = axs1[1, 2]
        if log["RL_thigh_vel"] != []:
            a.plot(time, log["RL_thigh_vel"], label="RL_thigh_vel")
        a.set(xlabel="time [s]", ylabel="RL_Thigh_Vel[rad/s]", title="RL_thighhigh_Velocity")
        a.legend()

        # 力矩 速度
        a = axs1[2, 2]
        if log["RL_thigh_vel"] != [] and log["RL_thigh_torque"] != []:
            a.plot(log["RL_thigh_vel"], log["RL_thigh_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="RL Torque/velocity curves")
        a.legend()

        # 力矩 时间
        a = axs1[0, 3]
        if log["RR_thigh_torque"] != []:
            a.plot(time, log["RR_thigh_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="RR_Thigh_Torque [Nm]", title="RR_Thigh_Torque")
        a.legend()

        # 速度 时间
        a = axs1[1, 3]
        if log["RR_thigh_vel"] != []:
            a.plot(time, log["RR_thigh_vel"], label="RR_thigh_vel")
        a.set(xlabel="time [s]", ylabel="RR_Thigh_Vel[rad/s]", title="RR_Thigh_Velocity")
        a.legend()
        
        # 力矩 速度
        a = axs1[2, 3]
        if log["RR_thigh_vel"] != [] and log["RR_thigh_torque"] != []:
            a.plot(log["RR_thigh_vel"], log["RR_thigh_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="RR Torque/velocity curves")
        a.legend()
        plt.tight_layout()

        ######图二 calf torque/vel#######
        
        # 力矩 时间
        a = axs2[0, 0]
        if log["FL_calf_torque"] != []:
            a.plot(time, log["FL_calf_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="FL_Calf_Torque [Nm]", title="FL_Calf_Torque")
        a.legend()

        # 速度 时间
        a = axs2[1, 0]
        if log["FL_calf_vel"] != []:
            a.plot(time, log["FL_calf_vel"], label="FL_calf_vel")
        a.set(xlabel="time [s]", ylabel="FL_Calf_Vel[rad/s]", title="FL_Calf_Velocity")
        a.legend()

        # 力矩 速度
        a = axs2[2, 0]
        if log["FL_calf_vel"] != [] and log["FL_calf_torque"] != []:
            a.plot(log["FL_calf_vel"], log["FL_calf_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="FL Torque/velocity curves")
        a.legend()

        # 力矩 时间
        a = axs2[0, 1]
        if log["FR_calf_torque"] != []:
            a.plot(time, log["FR_calf_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="FR_Calf_Torque [Nm]", title="FR_Calf_Torque")
        a.legend()

        # 速度 时间
        a = axs2[1, 1]
        if log["FR_calf_vel"] != []:
            a.plot(time, log["FR_calf_vel"], label="FR_calf_vel")
        a.set(xlabel="time [s]", ylabel="FR_Calf_Vel[rad/s]", title="FR_Calf_Velocity")
        a.legend()

        # 力矩 速度
        a = axs2[2, 1]
        if log["FR_calf_vel"] != [] and log["FR_calf_torque"] != []:
            a.plot(log["FR_calf_vel"], log["FR_calf_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="FR Torque/velocity curves")
        a.legend()

        # 力矩 时间
        a = axs2[0, 2]
        if log["RL_calf_torque"] != []:
            a.plot(time, log["RL_calf_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="RL_Calf_Torque [Nm]", title="RL_Calf_Torque")
        a.legend()

        # 速度 时间
        a = axs2[1, 2]
        if log["RL_calf_vel"] != []:
            a.plot(time, log["RL_calf_vel"], label="RL_calf_vel")
        a.set(xlabel="time [s]", ylabel="RL_Calf_Vel[rad/s]", title="RL_Calf_Velocity")
        a.legend()

        # 力矩 速度
        a = axs2[2, 2]
        if log["RL_calf_vel"] != [] and log["RL_calf_torque"] != []:
            a.plot(log["RL_calf_vel"], log["RL_calf_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="RL Torque/velocity curves")
        a.legend()

        # 力矩 时间
        a = axs2[0, 3]
        if log["RR_calf_torque"] != []:
            a.plot(time, log["RR_calf_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="RR_Calf_Torque [Nm]", title="RR_Calf_Torque")
        a.legend()

        # 速度 时间
        a = axs2[1, 3]
        if log["RR_calf_vel"] != []:
            a.plot(time, log["RR_calf_vel"], label="RR_calf_vel")
        a.set(xlabel="time [s]", ylabel="RR_Calf_Vel[rad/s]", title="RR_Calf_Velocity")
        a.legend()
        
        # 力矩 速度
        a = axs2[2, 3]
        if log["RR_calf_vel"] != [] and log["RR_calf_torque"] != []:
            a.plot(log["RR_calf_vel"], log["RR_calf_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="RR Torque/velocity curves")
        a.legend()
        plt.tight_layout()

        ######图三 wheel torque/vel#######
        
        # 力矩 时间
        a = axs3[0, 0]
        if log["FL_wheel_torque"] != []:
            a.plot(time, log["FL_wheel_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="FL_Wheel_Torque [Nm]", title="FL_Wheel_Torque")
        a.legend()

        # 速度 时间
        a = axs3[1, 0]
        if log["FL_wheel_vel"] != []:
            a.plot(time, log["FL_wheel_vel"], label="FL_wheel_vel")
            # a.plot(time, log["FL_wheel_ref_vel"], label="FL_wheel_ref_vel")
        a.set(xlabel="time [s]", ylabel="FL_Wheel_Vel[rad/s]", title="FL_Wheel_Velocity")
        a.legend()

        # 力矩 速度
        a = axs3[2, 0]
        if log["FL_wheel_vel"] != [] and log["FL_wheel_torque"] != []:
            a.plot(log["FL_wheel_vel"], log["FL_wheel_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="FL Torque/velocity curves")
        a.legend()

        # 力矩 时间
        a = axs3[0, 1]
        if log["FR_wheel_torque"] != []:
            a.plot(time, log["FR_wheel_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="FR_Wheel_Torque [Nm]", title="FR_Wheel_Torque")
        a.legend()

        # 速度 时间
        a = axs3[1, 1]
        if log["FR_wheel_vel"] != []:
            a.plot(time, log["FR_wheel_vel"], label="FR_wheel_vel")
            # a.plot(time, log["FR_wheel_ref_vel"], label="FR_wheel_ref_vel")
        a.set(xlabel="time [s]", ylabel="FR_Wheel_Vel[rad/s]", title="FR_Wheel_Velocity")
        a.legend()

        # 力矩 速度
        a = axs3[2, 1]
        if log["FR_wheel_vel"] != [] and log["FR_wheel_torque"] != []:
            a.plot(log["FR_wheel_vel"], log["FR_wheel_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="FR Torque/velocity curves")
        a.legend()

        # 力矩 时间
        a = axs3[0, 2]
        if log["RL_wheel_torque"] != []:
            a.plot(time, log["RL_wheel_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="RL_Wheel_Torque [Nm]", title="RL_Wheel_Torque")
        a.legend()

        # 速度 时间
        a = axs3[1, 2]
        if log["RL_wheel_vel"] != []:
            a.plot(time, log["RL_wheel_vel"], label="RL_wheel_vel")
            # a.plot(time, log["RL_wheel_ref_vel"], label="RL_wheel_ref_vel")
        a.set(xlabel="time [s]", ylabel="RL_Wheel_Vel[rad/s]", title="RL_Wheel_Velocity")
        a.legend()

        # 力矩 速度
        a = axs3[2, 2]
        if log["RL_wheel_vel"] != [] and log["RL_wheel_torque"] != []:
            a.plot(log["RL_wheel_vel"], log["RL_wheel_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="RL Torque/velocity curves")
        a.legend()

        # 力矩 时间
        a = axs3[0, 3]
        if log["RR_wheel_torque"] != []:
            a.plot(time, log["RR_wheel_torque"], label="real")
        a.set(xlabel="time [s]", ylabel="RR_Wheel_Torque [Nm]", title="RR_Wheel_Torque")
        a.legend()

        # 速度 时间
        a = axs3[1, 3]
        if log["RR_wheel_vel"] != []:
            a.plot(time, log["RR_wheel_vel"], label="RR_wheel_vel")
            # a.plot(time, log["RR_wheel_ref_vel"], label="RR_wheel_ref_vel")
        a.set(xlabel="time [s]", ylabel="RR_Wheel_Vel[rad/s]", title="RR_Wheel_Velocity")
        a.legend()
        
        # 力矩 速度
        a = axs3[2, 3]
        if log["RR_wheel_vel"] != [] and log["RR_wheel_torque"] != []:
            a.plot(log["RR_wheel_vel"], log["RR_wheel_torque"], label="real")
        a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="RR Torque/velocity curves")
        a.legend()
        plt.tight_layout()

        ######图四 FL跟踪#######
        
        # 位置 跟踪位置 时间
        a = axs4[0, 0]
        if log["FL_hip_pos"] != []:
            a.plot(time, log["FL_hip_pos"], label="FL_hip_pos")
            # a.plot(time, log["left_hip_action"], label="left hip action")
            # a.plot(time, log["FL_hip_ref_pos"], label="FL_hip_ref_pos")  
        a.set(xlabel="time [s]", ylabel="FL_Hip_Pos[rad]", title="FL_Hip_Pos")
        a.legend()

        # 位置 跟踪位置 时间
        a = axs4[0, 1]
        if log["FL_thigh_pos"] != []:
            a.plot(time, log["FL_thigh_pos"], label="FL_thigh_pos")
            # a.plot(time, log["left_hip_action"], label="left hip action")
            # a.plot(time, log["FL_thigh_ref_pos"], label="FL_thigh_ref_pos")  
        a.set(xlabel="time [s]", ylabel="FL_Thigh_Pos[rad]", title="FL_Thigh_Pos")
        a.legend()
               
        # 位置 跟踪位置 时间
        a = axs4[1, 0]
        if log["FL_calf_pos"] != []:
            a.plot(time, log["FL_calf_pos"], label="FL_calf_pos")
            # a.plot(time, log["left_hip_action"], label="left hip action")
            # a.plot(time, log["FL_calf_ref_pos"], label="FL_calf_ref_pos")  
        a.set(xlabel="time [s]", ylabel="FL_Calf_Pos[rad]", title="FL_Calf_Pos")
        a.legend()
                       
        # 速度 跟踪速度 时间
        a = axs4[1, 1]
        if log["FL_wheel_vel"] != []:
            a.plot(time, log["FL_wheel_vel"], label="FL_wheel_vel")
            # a.plot(time, log["left_hip_action"], label="left hip action")
            # a.plot(time, log["FL_wheel_ref_vel"], label="FL_wheel_ref_vel")  
        a.set(xlabel="time [s]", ylabel="FL_Wheel_Pos[rad]", title="FL_Wheel_Pos")
        a.legend()

        ######图五 FR跟踪#######
        
        # 位置 跟踪位置 时间
        a = axs5[0, 0]
        if log["FR_hip_pos"] != []:
            a.plot(time, log["FR_hip_pos"], label="FR_hip_pos")
            # a.plot(time, log["left_hip_action"], label="left hip action")
            # a.plot(time, log["FR_hip_ref_pos"], label="FR_hip_ref_pos")  
        a.set(xlabel="time [s]", ylabel="FR_Hip_Pos[rad]", title="FR_Hip_Pos")
        a.legend()

        # 位置 跟踪位置 时间
        a = axs5[0, 1]
        if log["FR_thigh_pos"] != []:
            a.plot(time, log["FR_thigh_pos"], label="FR_thigh_pos")
            # a.plot(time, log["left_hip_action"], label="left hip action")
            # a.plot(time, log["FR_thigh_ref_pos"], label="FR_thigh_ref_pos")  
        a.set(xlabel="time [s]", ylabel="FR_Thigh_Pos[rad]", title="FR_Thigh_Pos")
        a.legend()
               
        # 位置 跟踪位置 时间
        a = axs5[1, 0]
        if log["FR_calf_pos"] != []:
            a.plot(time, log["FR_calf_pos"], label="FR_calf_pos")
            # a.plot(time, log["left_hip_action"], label="left hip action")
            # a.plot(time, log["FR_calf_ref_pos"], label="FR_calf_ref_pos")  
        a.set(xlabel="time [s]", ylabel="FR_Calf_Pos[rad]", title="FR_Calf_Pos")
        a.legend()
                       
        # 速度 跟踪速度 时间
        a = axs5[1, 1]
        if log["FR_wheel_vel"] != []:
            a.plot(time, log["FR_wheel_vel"], label="FR_wheel_vel")
            # a.plot(time, log["left_hip_action"], label="left hip action")
            # a.plot(time, log["FR_wheel_ref_vel"], label="FR_wheel_ref_vel")  
        a.set(xlabel="time [s]", ylabel="FR_Wheel_Pos[rad]", title="FR_Wheel_Pos")
        a.legend()

        ######图六 RL跟踪#######
        
        # 位置 跟踪位置 时间
        a = axs6[0, 0]
        if log["RL_hip_pos"] != []:
            a.plot(time, log["RL_hip_pos"], label="RL_hip_pos")
            # a.plot(time, log["left_hip_action"], label="left hip action")
            # a.plot(time, log["RL_hip_ref_pos"], label="RL_hip_ref_pos")  
        a.set(xlabel="time [s]", ylabel="RL_Hip_Pos[rad]", title="RL_Hip_Pos")
        a.legend()

        # 位置 跟踪位置 时间
        a = axs6[0, 1]
        if log["RL_thigh_pos"] != []:
            a.plot(time, log["RL_thigh_pos"], label="RL_thigh_pos")
            # a.plot(time, log["left_hip_action"], label="left hip action")
            # a.plot(time, log["RL_thigh_ref_pos"], label="RL_thigh_ref_pos")  
        a.set(xlabel="time [s]", ylabel="RL_Thigh_Pos[rad]", title="RL_Thigh_Pos")
        a.legend()
               
        # 位置 跟踪位置 时间
        a = axs6[1, 0]
        if log["RL_calf_pos"] != []:
            a.plot(time, log["RL_calf_pos"], label="RL_calf_pos")
            # a.plot(time, log["left_hip_action"], label="left hip action")
            # a.plot(time, log["RL_calf_ref_pos"], label="RL_calf_ref_pos")  
        a.set(xlabel="time [s]", ylabel="RL_Calf_Pos[rad]", title="RL_Calf_Pos")
        a.legend()
                       
        # 速度 跟踪速度 时间
        a = axs6[1, 1]
        if log["RL_wheel_vel"] != []:
            a.plot(time, log["RL_wheel_vel"], label="RL_wheel_vel")
            # a.plot(time, log["left_hip_action"], label="left hip action")
            # a.plot(time, log["RL_wheel_ref_vel"], label="RL_wheel_ref_vel")  
        a.set(xlabel="time [s]", ylabel="RL_Wheel_Vel[rad]", title="RL_Wheel_Vel")
        a.legend()

        ######图七 RR跟踪#######
        
        # 位置 跟踪位置 时间
        a = axs7[0, 0]
        if log["RR_hip_pos"] != []:
            a.plot(time, log["RR_hip_pos"], label="RR_hip_pos")
            # a.plot(time, log["left_hip_action"], label="left hip action")
            # a.plot(time, log["RR_hip_ref_pos"], label="RR_hip_ref_pos")  
        a.set(xlabel="time [s]", ylabel="RR_Hip_Pos[rad]", title="RR_Hip_Pos")
        a.legend()

        # 位置 跟踪位置 时间
        a = axs7[0, 1]
        if log["RR_thigh_pos"] != []:
            a.plot(time, log["RR_thigh_pos"], label="RR_thigh_pos")
            # a.plot(time, log["left_hip_action"], label="left hip action")
            # a.plot(time, log["RR_thigh_ref_pos"], label="RR_thigh_ref_pos")  
        a.set(xlabel="time [s]", ylabel="RR_Thigh_Pos[rad]", title="RR_Thigh_Pos")
        a.legend()
               
        # 位置 跟踪位置 时间
        a = axs7[1, 0]
        if log["RR_calf_pos"] != []:
            a.plot(time, log["RR_calf_pos"], label="RR_calf_pos")
            # a.plot(time, log["left_hip_action"], label="left hip action")
            # a.plot(time, log["RR_calf_ref_pos"], label="RR_calf_ref_pos")  
        a.set(xlabel="time [s]", ylabel="RR_Calf_Pos[rad]", title="RR_Calf_Pos")
        a.legend()
                       
        # 速度 跟踪速度 时间
        a = axs7[1, 1]
        if log["RR_wheel_vel"] != []:
            a.plot(time, log["RR_wheel_vel"], label="RR_wheel_vel")
            # a.plot(time, log["left_hip_action"], label="left hip action")
            # a.plot(time, log["RR_wheel_ref_vel"], label="RR_wheel_ref_vel")  
        a.set(xlabel="time [s]", ylabel="RR_Wheel_Vel[rad]", title="RR_Wheel_Vel")
        a.legend()

                # # pitch 
        a = axs8[0, 0]
        if log["base_pitch"]:
            a.plot(time, log["base_pitch"], label="pitch")
        if log["base_roll"]:
            a.plot(time, log["base_roll"], label="roll")
        if log["base_yaw"]:
            a.plot(time, log["base_yaw"], label="yaw")
        a.set(xlabel="time [s]", ylabel="ang [rad]", title="IMU ang")
        a.legend()






        # # 图一 knee
        # # 力矩 时间
        # a = axs1[0, 0]
        # if log["left_knee_torque"] != []:
        #     a.plot(time, log["left_knee_torque"], label="real")
        # a.set(xlabel="time [s]", ylabel="Left knee Torque [Nm]", title="Left knee Torque")
        # a.legend()
        # # 速度 时间
        # a = axs1[1, 0]
        # if log["left_knee_vel"] != []:
        #     a.plot(time, log["left_knee_vel"], label="knee vel")
        # a.set(xlabel="time [s]", ylabel="Left knee Vel[rad/s]", title="Left knee Velocity")
        # a.legend()
        # # 力矩 速度
        # a = axs1[0, 2]
        # if log["left_knee_vel"] != [] and log["left_knee_torque"] != []:
        #     a.plot(log["left_knee_vel"], log["left_knee_torque"], label="real")
        # a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="Torque/velocity curves")
        # a.legend()

        # a = axs1[0, 1]
        # if log["right_knee_torque"] != []:
        #     a.plot(time, log["right_knee_torque"], label="real")
        # a.set(xlabel="time [s]", ylabel="right knee Torque [Nm]", title="right knee Torque")
        # a.legend()
        # # 速度 时间
        # a = axs1[1, 1]
        # if log["right_knee_vel"] != []:
        #     a.plot(time, log["right_knee_vel"], label="knee vel")
        # a.set(xlabel="time [s]", ylabel="right knee Vel[rad/s]", title="right knee Velocity")
        # a.legend()
        # # 力矩 速度
        # a = axs1[1, 2]
        # if log["right_knee_vel"] != [] and log["right_knee_torque"] != []:
        #     a.plot(log["right_knee_vel"], log["right_knee_torque"], label="real")
        # a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="Torque/velocity curves")
        # a.legend()
        # # 角度 时间
        # a = axs1[2, 0]
        # if log["left_knee_pos"] != []:
        #     a.plot(time, log["left_knee_pos"], label="left knee pos")
        #     # a.plot(time, log["left_knee_action"], label="left knee action")
        #     a.plot(time, log["left_knee_ref_pos"], label="left knee ref pos") ###xlh
        # a.set(xlabel="time [s]", ylabel="left knee pos[rad]", title="Left knee Pos")
        # a.legend()
        # a = axs1[2, 1]
        # if log["right_knee_pos"] != []:
        #     a.plot(time, log["right_knee_pos"], label="right knee pos")
        #     # a.plot(time, log["right_knee_action"], label="right knee action")
        #     a.plot(time, log["right_knee_ref_pos"], label="right knee ref pos") ###xlh
        # a.set(xlabel="time [s]", ylabel="right knee pos[rad]", title="Right knee Pos")
        # a.legend()

        # # 图二 wheel
        # # 力矩 时间
        # a = axs2[0, 0]
        # if log["left_wheel_torque"] != []:
        #     a.plot(time, log["left_wheel_torque"], label="real")
        # a.set(xlabel="time [s]", ylabel="Left wheel Torque [Nm]", title="Left wheel Torque")
        # a.legend()
        # # 速度 时间
        # a = axs2[1, 0]
        # if log["left_wheel_vel"] != []:
        #     a.plot(time, log["left_wheel_vel"], label="wheel vel")
        #     # a.plot(time, log["left_wheel_action"], label="wheel action")
        #     a.plot(time, log["left_wheel_ref_vel"], label="wheel ref vel") ###xlh
        # a.set(xlabel="time [s]", ylabel="Left wheel Vel[rad/s]", title="Left wheel Velocity")
        # a.legend()
        # # 力矩 速度
        # a = axs2[0, 2]
        # if log["left_wheel_vel"] != [] and log["left_wheel_torque"] != []:
        #     a.plot(log["left_wheel_vel"], log["left_wheel_torque"], label="real")
        # a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="Torque/velocity curves")
        # a.legend()

        # a = axs2[0, 1]
        # if log["right_wheel_torque"] != []:
        #     a.plot(time, log["right_wheel_torque"], label="real")
        # a.set(xlabel="time [s]", ylabel="right wheel Torque [Nm]", title="right wheel Torque")
        # a.legend()
        # # 速度 时间
        # a = axs2[1, 1]
        # if log["right_wheel_vel"] != []:
        #     a.plot(time, log["right_wheel_vel"], label="wheel vel")
        #     # a.plot(time, log["right_wheel_action"], label="wheel action")
        #     a.plot(time, log["right_wheel_ref_vel"], label="wheel ref vel") ###xlh
        # a.set(xlabel="time [s]", ylabel="right wheel Vel[rad/s]", title="right wheel Velocity")
        # a.legend()
        # # 力矩 速度
        # a = axs2[1, 2]
        # if log["right_wheel_vel"] != [] and log["right_wheel_torque"] != []:
        #     a.plot(log["right_wheel_vel"], log["right_wheel_torque"], label="real")
        # a.set(xlabel="Joint vel [rad/s]", ylabel="Joint Torque [Nm]", title="Torque/velocity curves")
        # a.legend()
        # # 角度 时间 
        # a = axs2[2, 0]
        # if log["left_wheel_pos"] != []:
        #     a.plot(time, log["left_wheel_pos"], label="left wheel pos")
        # a.set(xlabel="time [s]", ylabel="left wheel pos[rad]", title="Left wheel Pos")
        # a.legend()
        # a = axs2[2, 1]
        # if log["right_wheel_pos"] != []:
        #     a.plot(time, log["right_wheel_pos"], label="right wheel pos")
        # a.set(xlabel="time [s]", ylabel="right wheel pos[rad]", title="Right wheel Pos")
        # a.legend()

        # # # 图三
        # # # 速度跟随
        # a = axs3[0, 0]
        # if log["base_vel_x"]:
        #     a.plot(time, log["base_vel_x"], label="real")
        # if log["est_lin_vel_x"]:
        #     a.plot(time, log["est_lin_vel_x"], label="est")
        # if log["command_vel_x"]:
        #     a.plot(time, log["command_vel_x"], label="commanded")
        # a.set(xlabel="time [s]", ylabel="base lin vel [m/s]", title="Base velocity x")
        # a.legend()

        # # 角速度跟随
        # a = axs3[0, 1]
        # if log["base_ang_vel"]:
        #     a.plot(time, log["base_ang_vel"], label="real")
        # if log["est_ang_vel"]:
        #     a.plot(time, log["est_lin_vel_x"], label="est")
        # if log["command_ang_vel"]:
        #     a.plot(time, log["command_ang_vel"], label="commanded")
        # a.set(xlabel="time [s]", ylabel="base ang vel [rad/s]", title="Base ang vel")
        # a.legend()

        # # 高度跟随
        # a = axs3[1, 0]
        # if log["base_height"]:
        #     a.plot(time, log["base_height"], label="real")
        # if log["est_ang_vel"]:
        #     a.plot(time, log["est_lin_vel_x"], label="est")
        # if log["command_height"]:
        #     a.plot(time, log["command_height"], label="commanded")
        # a.set(xlabel="time [s]", ylabel="base height [m]", title="Base height")
        # a.legend()

        # # pitch 
        # a = axs3[1, 1]
        # if log["base_pitch"]:
        #     a.plot(time, log["base_pitch"], label="pitch")
        # if log["base_roll"]:
        #     a.plot(time, log["base_roll"], label="roll")
        # if log["base_yaw"]:
        #     a.plot(time, log["base_yaw"], label="yaw")
        # a.set(xlabel="time [s]", ylabel="ang [rad]", title="IMU ang")
        # a.legend()

        plt.tight_layout()
        plt.show()

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")

    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()
