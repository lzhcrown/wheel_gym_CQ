第 4.5.1 节：显式状态信息与估计器结构对照实验

一、六组入口
所有配置均通过 common.py 继承当前 Nezha2Cfg / Nezha2CfgPPO，
并使用同一个 Chapter451Env 环境实现。

任务名                      配置文件                 Actor 输入  Critic 输入
exp451_baseline              baseline.py                 130        1113
exp451_critic_truth          critic_truth.py             130        1125
exp451_actor_critic_truth    actor_critic_truth.py       138        1125
exp451_est_128_64            est_128_64.py               138        1125
exp451_est_64_64             est_64_64.py                138        1125
exp451_est_64_32             est_64_32.py                138        1125

baseline：Actor 使用两帧常规观测；Critic 不显式加入线速度和高度。
critic_truth：仅 Critic 每帧加入真实 vx、vy、vz、h。
actor_critic_truth：Actor 和 Critic 每帧都加入真实 vx、vy、vz、h。
三个 est 组：Critic 使用真值，Actor 使用估计值；只改变估计器隐藏层宽度。

估计器输入为两帧 65 维常规观测（共 130 维），输出 8 维：
[vx(t-1), vy(t-1), vz(t-1), h(t-1), vx(t), vy(t), vz(t), h(t)]。
Actor 按 [65 维上一帧, 4 维上一帧估计, 65 维当前帧, 4 维当前帧估计]
拼成 138 维。监督目标取 Critic 三帧历史中的最后两帧，各取末尾 4 维。
两个时刻分别预测、分别监督；训练与推理使用相同顺序。
真值与估计值使用相同尺度：线速度乘 2，高度乘 5。

二、统一条件
机器人统一使用 resources/robots/urdf2.1/urdf/nezha3_description.urdf，
来源记录在 resources/model_source.json。
奖励函数和权重、控制器、仿真、地形、指令、噪声、域随机化以及 PPO 参数
均使用当前框架配置。公共设置在 common.py 中集中继承，组文件仅写实验差异。
当前 feet_air_time 权重为 4.0；未恢复历史日志中的 1.0。
默认 seed=10、num_envs=4096、每次迭代采样 24 步，Actor/Critic 隐藏层均为
[512, 256, 128]；默认 max_iterations=1000001、save_interval=100。
三个估计器组统一使用学习率 0.001、每次 PPO 小批次后的监督更新次数 1。
若需调整共同训练预算、种子或并行环境数，请对六组使用相同命令行参数。

三、训练
项目内已创建 Python 3.8.20 的 .venv；安装与重建说明见项目根目录 TRAINING_ENV.txt。
依次训练六组，脚本自动激活环境：
cd /home/crown/wheel_gym_CQ
bash scripts/train_chapter451.sh --headless

六组均默认启用 headless，无仿真窗口；可像上面这样显式传入 --headless。

批量脚本统一使用 seed=10、num_envs=4096、max_iterations=10000、save_interval=100。
10000 是批量脚本显式指定的预算，公共配置文件中的默认值不变。
六组顺序为上表顺序。一组正常退出后再启动下一组；报错或 Ctrl+C 会停止整个批次。
既有数据保留，每次重新运行批量脚本会从第一组开始新训练。

修改所有组的共同预算，或仅预览命令：
bash scripts/train_chapter451.sh --max_iterations 20000 --num_envs 4096 --seed 10 --save_interval 100
bash scripts/train_chapter451.sh --dry-run

也可单独启动一组：
source scripts/activate_train.sh
python wheel_legged_gym/scripts/train.py --task exp451_est_64_64 --headless --seed 10 --num_envs 4096 --max_iterations 10000 --save_interval 100

无参数启动 train.py 时默认使用 exp451_baseline；命令行设置不再被脚本覆盖。
正式对照实验如采用多个随机种子，各组应使用同一组种子与相同迭代预算。
恢复训练示例（RUN_NAME 为对应日志目录名）：
python wheel_legged_gym/scripts/train.py --task exp451_est_64_64 --resume --load_run RUN_NAME --checkpoint 1000 --max_iterations 9000 --headless
恢复后 max_iterations 表示额外训练次数。新模型检查点编号代表已完成的迭代数。
自动恢复只搜索同一 task 的检查点，避免读到其他实验组的模型。

四、CSV 输出
每次运行写入 logs/chapter4_5_1/时间戳_组名/：
  events.out.tfevents.*    原始 TensorBoard 日志
  csv/*.csv               全部标量指标的逐条原始记录，训练时同步写入
  csv/tags.json           指标名与 CSV 文件名的对应表
  cli_args.json           实际命令行参数，含仿真设备与线程覆盖设置
  env_cfg.json            应用命令行覆盖后、环境初始化前的环境参数
  train_cfg.json          实际训练参数
  run_info.json           task、种子、实际 Actor/Critic 输入维数、URDF 哈希
  source/                 本次训练的 Python 源代码快照
  source_sha256.json      源码哈希
  model_source.json       机器人资源来源清单
  model_N.pt              已完成 N 次迭代的模型及优化器状态

CSV 列为 wall_time,step,value；原始数值不做平滑。
Train/mean_reward 保留当前框架的统计口径：最近最多 100 个已结束回合的平均回报。
尚无回合结束时，该指标不会产生数据行。其横轴为从 0 开始的训练迭代号。
所有奖励分项和损失均保存在 CSV 中，不自动绘图、不平滑、不汇总统计。
训练结束或正常 Ctrl+C 后会关闭日志写入器，保留已经记录的数据。
后续可直接在 MATLAB 中读取 CSV，自行选取相同迭代区间并绘图。

示例文件：csv/Train_mean_reward.csv、csv/Train_mean_episode_length.csv；
完整指标名与文件名的映射以 csv/tags.json 为准。

单次训练补导出全部指标（覆盖该运行目录内对应的 CSV）：
python wheel_legged_gym/scripts/tensorboard_data_ex.py logs/chapter4_5_1/RUN_NAME

批量补导出：将各个实际运行目录依次作为位置参数传入，例如：
python wheel_legged_gym/scripts/tensorboard_data_ex.py logs/chapter4_5_1/RUN1 logs/chapter4_5_1/RUN2 logs/chapter4_5_1/RUN3 logs/chapter4_5_1/RUN4 logs/chapter4_5_1/RUN5 logs/chapter4_5_1/RUN6

每组 CSV 都保存在各自运行目录的 csv/ 中，不合并或平均不同实验组的数据。
导出脚本仅需 TensorBoard，不需要 Isaac Gym 或绘图库。
训练被强制终止（如 kill -9）后，可用导出脚本恢复 TensorBoard 已落盘的指标。

五、回归检查
python -m unittest discover -s tests -v
测试覆盖公共参数一致性、配置隔离、两帧真值和估计输入对齐、六组 CPU PPO
更新及检查点恢复，以及 CSV 原始数值、单组及批量导出且不生成图片；测试输出写入临时目录。
