门控模态双编码器：五组实验依次重新训练

在项目根目录执行：
cd /home/bit/wheel_gym_CQ
bash scripts/train_mine_ablation.sh --max_iterations 10000 --num_envs 4096 --save_interval 200 --seed 10 --headless

每组使用相同种子 10，从头训练 10000 次迭代，共 5 次训练。
脚本自动激活项目的 Python 3.8 .venv，一个实验完成后才启动下一个：
1. full：完整的门控模态双编码器方法。
2. no_latent_input：去掉 Actor 的隐特征输入。
3. no_gate：去掉门控融合，同时关闭关联模态辅助损失。
4. no_state_input：去掉 Actor 的显式状态估计输入。
5. no_mode_loss：保留门控，仅去掉模态辅助损失。

每次训练使用 4096 个并行环境，每 200 次迭代保存模型。
任一实验报错或手动中断后，脚本停止，不继续启动后续实验。
重新执行命令会从第一组开始新的训练，不自动恢复旧检查点。
本命令显式使用 --seed 10；省略该选项会采用脚本默认的三个种子，共训练 15 次。

当前初始姿态与 wheel_gym_lzh 的 Nezha2 配置一致。
每条腿按髋关节、大腿、小腿、轮关节顺序，单位为弧度：
右前 FR、右后 RR： 0.1, 0.925, -1.85, 0.0
左前 FL、左后 RL：-0.1, 0.925, -1.85, 0.0
训练出生高度：init_state.pos[2] = 0.6 米，相对于地形环境原点。
高度奖励目标：rewards.base_height_target = 0.5 米，相对于采样地面的平均高度。
五组继承同一套环境、奖励和训练设置，仅改变对应消融开关。
训练阶梯踏面宽度：terrain.stair_tread_width = 0.3 m，实际为3个10cm网格。
训练区域10×10m、中心平台4×4m，因此每侧10级；单级高度继续随课程变化，
名义值为3～19.2cm，按5mm网格量化后约3～19cm。五组共用该设置。


结果保存在 logs/mine_ablation/时间戳_实验名_seed10/，不同运行分别保存。
训练指标 CSV 仅输出平均奖励：csv/Train_mean_reward.csv。
CSV 列为 wall_time、step、value；step 为迭代号，value 为平均回合累计奖励。

只预览五条训练命令、不启动训练：
bash scripts/train_mine_ablation.sh --max_iterations 10000 --num_envs 4096 --save_interval 200 --seed 10 --headless --dry-run
