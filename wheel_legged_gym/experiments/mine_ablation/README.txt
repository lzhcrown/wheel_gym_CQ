门控模态双编码器：五组消融实验

一、组别与开关
任务名前缀为 mine_ablation_。开关位于训练配置的 policy 中。

变体                 use_latent_input  use_state_input  use_mode_gate  mode_loss_coef
full                 True              True             True           0.5
no_latent_input      False             True             True           0.5
no_gate              True              True             False          0.0
no_state_input       True              False            True           0.5
no_mode_loss         True              True             True           0.0

这五组均以 full 为参照，各自独立训练，不是按顺序累计删除模块，也不从
full 的已训练模型继续训练。配置文件分别与变体同名，公共配置位于 common.py。

full：完整方法，对应当前 nezha2_mine 的网络、环境及训练超参。
no_latent_input：将送入 Actor 的 16 维隐变量置零，编码器及辅助训练保留。
no_state_input：将送入 Actor 的 4 维估计状态置零，状态回归及其他辅助训练保留。
no_gate：使用未经过门控融合的归一化隐变量，同时停用依赖门控标签的模态损失。
no_mode_loss：保留门控融合，仅将模态损失权重从 0.5 改为 0。

no_gate 保留门控网络和模态原型的实例化，但不执行该分支，保证构造顺序及
随机初始化与其他组一致。这一组检验门控模态模块的作用；no_mode_loss 则
单独检验模态损失。它们不是同一个对照。

辅助目标为 L_state + L_swap + mode_loss_coef * L_mode。
no_mode_loss 仍保留状态回归 L_state 和双编码器对齐 L_swap；门控网络及
三个模态原型仍可通过 L_swap 更新。当前 L_mode 使用门控 argmax 生成标签，
对 z_s 使用 detach，因此该项直接更新的是模态原型，不直接向门控或 Source
反传。它没有使用人工地形类别标签，不应描述为带地形真值的门控分类监督。

二、控制变量
所有组使用同一 Nezha2 环境、奖励、噪声、随机化、地形及课程设置。
PPO、优化器、采样长度、学习率规则及网络隐藏层保持一致。
Source 与 Gate 的输入都是按旧到新排列的两帧观测，共 130 维。
Actor 输入维数固定为最新一帧 65 维 + 估计状态 4 维 + 隐变量 16 维 = 85。
Critic 为三帧共 1125 维；Target 编码当前动作前的单帧特权观测 p_t，共 375 维。
四维状态回归仍使用下一状态真值，终止回合使用重置前的终止状态。
门控启用时，采样、推理和辅助训练均逐模态归一化原型后再融合。

各组仍实例化全部网络，因此输入消融衡量的是信息输入的贡献，不是拆除
网络后比较参数量、计算量或速度。相同种子下各组的初始参数应逐项相同。
策略行为变化后，采样状态、课程进度和自适应学习率轨迹可以不同。

当前共同训练设置（CLI 覆盖项须对所有组一致）：
  环境数 4096；每轮每环境采样 24 步；回合上限 20 秒；
  Actor / Critic 隐藏层均为 [512, 256, 128]，ELU；
  Source 隐藏层 [256, 128, 64]，输出状态 4 维、隐特征 16 维；
  Target 隐藏层 [256, 128, 64]，输出隐特征 16 维；
  PPO 初始学习率 0.001，adaptive，desired_kl=0.01；
  每轮更新 5 epochs、4 mini-batches，clip=0.2，gamma=0.99，lambda=0.95；
  value_loss_coef=1.0，entropy_coef=0.01，PPO max_grad_norm=1.0；
  编码器辅助训练 max_grad_norm=10.0；初始动作噪声标准差 1.0。
奖励、控制、观测归一化、随机化、地形及课程均继承已确定的主方法配置。

三、顺序训练
cd /home/bit/wheel_gym_CQ
bash scripts/train_mine_ablation.sh --dry-run
bash scripts/train_mine_ablation.sh --headless

默认每次训练 10000 次迭代、4096 个环境、每 200 次迭代保存模型；
默认种子为 10、20、30，共 15 次独立训练。先依次运行 full 的三个种子，
再依次运行 no_latent_input、no_gate、no_state_input、no_mode_loss。
配置继承原方法的 max_iterations=100000，批量脚本用命令行统一覆盖为 10000。

统一修改预算或种子：
bash scripts/train_mine_ablation.sh --max_iterations 20000 --seeds 10,20,30
bash scripts/train_mine_ablation.sh --max_iterations 10000 --num_envs 4096 --save_interval 200 --seed 10
bash scripts/train_mine_ablation.sh --help

--seed 与 --seeds 不能同时使用，种子列表不能重复。批量脚本固定 task、
experiment_name 和 run_name，拒绝覆盖这些参数，以免日志分组与实际配置不符。
脚本自动激活 .venv 并启用 headless；任一任务失败或 Ctrl+C 后停止后续任务。
重新执行脚本会从第一组开始新训练，不自动续训，也不覆盖已有运行目录。

单组示例（显式写出种子及对应运行名）：
source scripts/activate_train.sh
python wheel_legged_gym/scripts/train.py --task mine_ablation_no_gate --headless --seed 10 --num_envs 4096 --max_iterations 10000 --save_interval 200 --run_name no_gate_seed10

四、训练记录
批量脚本每次输出至 logs/mine_ablation/时间戳_变体_seedN/。
唯一指标 CSV 为 csv/Train_mean_reward.csv，列名为 wall_time,step,value。
它实时记录最近最多 100 个已结束回合的平均累计回报；step 为从 0 开始的
训练迭代号。尚无回合结束时不产生该指标数据行，不额外平滑或绘图。
TensorBoard、模型、实际 CLI/环境/训练配置和源码快照继续保留。

论文指标定义与当前采集范围见同目录 EVALUATION_METRICS.txt。
独立评估采集器已实现，使用 scripts/evaluate_mine.sh；批量训练脚本仍仅
执行上述训练并记录平均回报。评估协议、CSV字段及命令详见
wheel_legged_gym/evaluation/README.txt。默认6.2×6.2米、7类地形，采用全部四轮的轮入/轮出
测量窗口；当前只保存指标公式涉及的基础数据，不预设VIR阈值或在线输出
指标结果。入块前的接近过程不计入采样窗口，各组需采用相同的评估协议。

五、Baseline 建议（设计方案，尚未加入上述五组任务或批量命令）
当前完整方法本身就使用 PPO 优化策略。建议基线命名为 PPO-MLP-2frame，
比较普通 MLP 表征与门控模态双编码器表征，而不是将二者称作不同的策略
优化算法。采用 legged_gym / rsl_rl 的普通 MLP Actor-Critic + PPO 结构思路，
机器人、环境、观测内容、奖励、训练超参仍沿用本项目的相同设置。

推荐主 Baseline：
  Actor：直接输入 [o_{t-1}, o_t] 共 130 维，经 [512, 256, 128] 输出 16 维动作；
  Critic：保持三帧特权观测共 1125 维和相同隐藏层；
  不使用 Source / Target、显式状态估计、隐特征、门控、原型或任何辅助损失；
  仅使用 PPO 的策略损失、价值损失及熵项，保持相同优化和采样设置。
完整方法通过 Source / Gate 使用两帧历史，因此基线同样获得两帧能控制
可用观测信息的范围；不向基线 Actor 额外提供真实速度、真实高度或高程图。
特权信息保留给 Critic，用于训练。不同架构不要求总参数量完全相同，比较
推理效率时应另报实际推理网络参数量和耗时。

若预算允许，可补充 PPO-MLP-1frame：Actor 只输入当前 65 维，其余相同。
它回答常规单帧策略的性能；仅与该基线比较，会同时包含历史信息和表征
结构两种差异。若只选一个主 Baseline，建议选择上述两帧版本。

实施时需使用普通 PPO 的 runner / rollout，而非仅把 MINE 的两个 Actor
输入置零后仍训练辅助网络。当前普通 PPO 未注册为可训练任务，独立评估
也只加载 MINE 模型，增加 Baseline 时需同时接入训练、推理、导出和评估。

所有方法使用相同训练种子（默认 10、20、30）、迭代数和环境交互预算，
采用相同模型选择规则（例如各次训练的最终 checkpoint）。固定评估地形、
命令、初始化规则及评估种子，分别计算每个训练种子的指标，再汇总均值和
离散程度。各组策略实际轨迹和成功次数不要求相同。结论应由平均回报和
固定地形上的通过表现、能耗及速度跟踪等数据共同支持，不预设 full 必然最好。

结构参考：
  https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py
  https://github.com/leggedrobotics/rsl_rl/blob/v1.0.2/rsl_rl/modules/actor_critic.py
  https://github.com/leggedrobotics/rsl_rl/blob/v1.0.2/rsl_rl/runners/on_policy_runner.py
