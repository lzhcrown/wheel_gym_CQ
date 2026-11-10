Isaac Gym 6.2×6.2米地形通过测试：公式输入数据采集
==========================================

当前模式只保存基础数据和测试事件，供之后离线计算指标。
不预设速度不稳定率阈值，不在线输出VIR、平均功率、能量、COT、RMSE或成功率。
保留成功/失败/超时等结果及数量，它们是之后计算成功率的输入。
训练CSV继续仅输出平均奖励；本入口单独载入训练好的策略进行仿真采集。

1. 测试窗口与地形
七类地形块统一6.2×6.2m；其中阶梯中心平台3×3m，每侧8级、踏面0.2m。
起步/退出平台在块外，各长1.5m；左右支撑各1m，完整行驶走廊为9.2×8.2m。
前进时首排前轮进入开始，末排后轮从对侧完全离开结束；倒车时前后交换。
前轮已离开、后轮尚未进入的间隔仍属于同一次测试，持续记录。
进入前的接近过程不进入运动/力矩采样CSV；未能进入的尝试仍有事件记录。
全部采集使用同一窗口，不丢弃窗口前若干秒、不按阈值筛掉运动样本。

车轮判定使用真实轮刚体姿态及URDF碰撞圆柱的世界XY投影范围，计入半径、
宽度和姿态。全部四轮必须都从入口进入，所有四轮尾缘均超过出口
后才成功；绕过、从侧面离开或退回入口不算成功。允许抬轮或跳越。
机器人在外部平台上沿区域中心线初始化，起点有±0.04m的XY扰动，姿态朝向
世界+x；反向测试从高x端倒车进入。6.2m宽区域覆盖整车轮迹，默认检查全部四轮。
实际轮子包络按所加载配置与URDF计算，不再采用旧40cm测试的单侧轮对齐。

进入/退出在策略步末检测，默认分辨率约0.02s。首轮入块被检测到的那一帧
作为t=0，之前区间不计入；末轮完全离开被检测到的那个控制步结束采集。
每个边界的检测相对几何事件可能滞后一个控制周期，不声称精确接触时刻。
四轮数据顺序为FL、FR、RL、RR；物理关节列的顺序另由metadata指定。

2. 运行命令
脚本自动激活现有Python3.8环境。先预览协议，无需模型、不创建仿真：
  bash scripts/evaluate_mine.sh --dry-run

训练完成后，使用对应任务最新检查点：
  bash scripts/evaluate_mine.sh --task mine_ablation_full --headless

正式比较建议固定模型，将YOUR_RUN_DIRECTORY替换为实际训练目录：
  bash scripts/evaluate_mine.sh --task mine_ablation_full --load_run YOUR_RUN_DIRECTORY --checkpoint 10000 --envs-per-terrain 32 --episodes-per-env 5 --seed 100 --headless

倒车另起一轮采集：
  bash scripts/evaluate_mine.sh --task mine_ablation_full --speed -1 --headless

--load_run为logs/mine_ablation/内的目录名；也可改用
--checkpoint-path /绝对路径/model_10000.pt，不能同时使用两套模型选择参数。
支持nezha2_mine，以及mine_ablation_后接full、no_latent_input、no_gate、
no_state_input、no_mode_loss。自动选择按task隔离，再取最新run/最大模型编号。
优先恢复训练run的env_cfg.json/train_cfg.json；缺少完整快照时仅显式
--allow-current-config可使用当前配置。权重严格加载不保证所有语义兼容。

主要选项：
  --size-m 6.2                七类标准测试区域均固定6.2×6.2m
  --envs-per-terrain N         每类并行槽数量，默认16
  --episodes-per-env N         每槽尝试数，默认1；别名--attempts-per-env
  --terrain-types ...         默认下述7类；可选其中一个或多个名称，用逗号分隔
  --speed V                   机体系vx，默认+1m/s；负值倒车，vy=wz=0
  --wheel-side all             当前只允许all，全部四轮通过
  --spawn-xy-jitter M          默认0.04m，最大0.05m
  --max-seconds T              整次尝试含接近阶段的上限，默认1m/s时20.4s
  --step-width 0.2            标准凹/凸阶梯踏面20cm，两侧各八级
  --rough-height M            粗糙地面起伏幅度参数，默认0.04m
  --slope-angle-deg DEG       坡面角度，默认12°
  --seed N                    评估地形/起点seed，默认10；与训练seed区分
  --output-dir PATH           新目录或空目录，拒绝覆盖既有结果
  --viewer                    显示窗口；默认headless
  --sim-device cpu             小规模CPU检查；默认cuda:0
  --trajectory-stride 1       仅允许1，完整记录每个控制步
  --help                      完整参数
--vir-threshold和--warmup-seconds已移除。原始采集不允许关闭/降采样控制记录，
否则不能完整重建阈值可调的VIR及全部时间段内的其他指标。
默认时限按2×(区域长度+1.5m)/|指令速度|+5s计算，基于两端平台中点之间
的路程；前进和倒车相同。可用--max-seconds显式覆盖，正式比较时各组统一。

3. 输出文件与采样频率
默认目录为evaluations/TASK/时间戳/，包含以下文件。

physics.csv：每个活跃环境、每个物理子步一行，默认约200Hz。
  physics_step、control_step、substep_index以及time_start_s/time_end_s/dt_s；
  torque_j00_nm ... torque_j15_nm：真正施加到仿真的16关节力矩，单位N·m；
  joint_vel_before_j00_radps ...：这个子步开始时的关节角速度，单位rad/s；
  joint_vel_after_j00_radps ...：这个子步结束时的关节角速度，单位rad/s。
  力矩是经过限幅/执行设置后的self.torques，速度来自真实dof_state。
  既非策略动作，也非归一化观测中的关节速度。逐关节保存正负号，不提前
  取绝对值或积分，因此可离线自行选择有符号功、绝对机械功或正机械功。
  关节列j00..j15与实际关节名称的映射见metadata.json的joint_columns。

trajectory.csv：每个活跃环境、每个控制步一行，默认约50Hz。
  当前与上一帧的世界XYZ位置、世界XYZ线速度；
  机体系XYZ线速度、XYZ角速度、当前区间生效的vx/vy/wz指令；
  世界姿态四元数xyzw、世界XYZ角速度；
  实际控制时间间隔、测量时间及样本编号；
  世界XYZ差分加速度，以及水平XY模长/XYZ模长。
  加速度由(current_world_velocity - previous_world_velocity)/control_dt计算，
  不是IMU比力，不滤波、不比较任何阈值、不产生“稳定/不稳定”分类。
  保存上一帧数据使首次差分可重建；所有行均可独立核对区间端点。
  不输出跟踪误差、功率、能量、COT、RMSE等聚合结果。

episodes.csv：每次通过尝试一行，记录是否实际开始测试、起止时间、状态、
原因、采样数量、全机质量、重力、目标方向及窗口初末位置/速度。
  window_started区分已进入与未进入；entry_time_s/exit_time_s相对本次初始化；
  elapsed_s是有采样的测量窗口时间，trial_elapsed_s包含接近时间。
  首轮进入前失败/超时的尝试，窗口动态数据为空、样本数0，不伪造零能耗。
  空窗口/同一帧完成等无法获得物理采样的情况也保留样本数量和标记。
  入块/终止帧轮投影、起点、区域边界等原始信息保存在metadata_json。
  wheel_contact_mask仅为轮投影重叠期间的净接触力诊断，不能定位接触点；
  不能据此认定接触发生在块上，成功判据也不强制要求接触。

summary.csv：只按地形/方向及overall统计事件数量：
  尝试数、完成数、success/failure/timeout/interrupted数量，
  started_tests/started_attempts以及未入块失败/超时等数量。
  不输出百分比、指标均值或标准差。之后按明确分母自行计算成功率。
  中断单独记录，不当作正常完成。并行起点不是独立模型训练seed。

metadata.json：完整配置、模型/资产/源码SHA256、实际关节列顺序与单位、
世界重力向量、全机质量、采样频率、窗口定义、几何参数、seed及运行状态。
各CSV均可通过env_id+episode_index关联同一次尝试；不能跨回合直接差分。
输出期间刷新CSV；Ctrl+C/可捕获异常会保存中断事件。kill -9或掉电不能保证
完成清理，metadata仍为running的目录不能当作完整试验。

4. 指标公式需要的数据映射（只说明后续用途，不在线计算）
成功率：episodes/summary中的成功数、已结束尝试数或已开始测试数。
  分母选全部发起尝试还是已进入窗口的测试需明确；未进入样本不能静默丢弃。
平均功率/机械能：physics中的逐关节tau、omega_before/after、dt，以及窗口时长。
  若采用与此前一致的梯形积分，子步机械功可在离线用
  sum_j[tau_j*(omega_before_j+omega_after_j)/2]*dt复算。
  abs/positive口径须先分别处理两端功率再积分，不能先正负抵消后取绝对值。
  数据不含电机电流、电压及效率模型，不能直接当电池电耗。
COT：上述力矩/角速度/时间数据，加全机质量M、重力g、世界位置序列。
  用位置序列自行确定XY路径、3D路径或沿目标方向净位移；路径基于控制步
  采样，是离散近似。不能把分母距离固定写成6.2m，首轮进入至末轮离开时
  整车还需走过自身轮距/轮径。距离接近0时，应另行定义无效样本规则。
速度不稳定率：每帧速度/差分加速度和实际dt。保留XYZ分量及模长，
  后续可自行选阈值、坐标分量、滤波和统计窗口，当前不作预设。
速度跟踪：机体系实际vx/vy、实际wz与同区间的对应指令、dt。
  之后可自行计算RMSE、MAE或其他误差定义，不混合m/s和rad/s的量纲。
不在采集阶段删除启动片段；若后续需要稳态片段，离线按时间戳统一筛选。

5. 几何与比较条件
新协议version=6，网格0.02m，竖直分辨率0.005m；块内有310×310网格间隔。
默认七类按以下顺序生成，每类16个并行槽，共112个；同类槽使用相同地形
几何，不同起点扰动，彼此关闭机器人碰撞。所有地形完整参数写入metadata。
  flat                    平地
  stairs_concave_15cm      凹式阶梯，每级下降/上升15cm，中心深度120cm
  stairs_concave_20cm      凹式阶梯，每级下降/上升20cm，中心深度160cm
  stairs_convex_15cm       凸式阶梯，每级上升/下降15cm，中心高度120cm
  stairs_convex_20cm       凸式阶梯，每级上升/下降20cm，中心高度160cm
  rough                   粗糙地面，随机起伏幅度参数±4cm
  slope                   沿+x上升12°的坡面，反向时下坡
凹凸是同心方形阶梯，踏面采用用户指定15～20cm范围中的20cm。
每侧八级，中心为3×3m平台；6.2m=8×0.2m+3m+8×0.2m。
以距四边的最小距离d决定高度：外圈d<0.2m为0，每深入0.2m升/降一级，
中心d≥1.6m为±八级。穿越中心线时依次经历0、±h、…、±8h、…、±h、0，
即一侧八次、另一侧八次，共十六次升降。15/20cm表示每级高度，
非整块地形总深度/高度；三维图中的阶梯是方形环绕结构，非仅沿x挤出的直梯。
--step-height仅用于旧region布局，标准patch通过四个独立名称固定梯级高度，
保证同一次采集保留两种高度。未启用离散障碍块，也不接受其CLI类型名。
坡面高差约6.2*tan(12°)=1.318m（5mm网格量化后约1.32m）；出口平台保持坡顶高度，不回落为零，避免
增加一个额外下落台阶。反向测试在高端平台初始化。为支撑整车，坡面同一
纵向高度剖面延伸至8.2m宽走廊；计时/判定仍只用中心6.2×6.2m区域。
粗糙地形在区域边界接回平地，实际网格高度受插值和5mm量化影响。
所有组使用相同地形、起点、侧别、指令、时限；关闭课程和训练随机扰动，
包括0.8~1.1随机力矩缩放。原有训练默认行为保持。
这是一次有限区域通过测试；长期稳态效率应另设连续地形试验。
新采集文件不覆盖以前已经生成的指标结果；两种文件结构应区分使用。

根据实际地形网格生成三维总览和中心剖面（无需模型、不运行物理仿真）：
  source scripts/activate_train.sh
  python scripts/plot_evaluation_terrain.py
图像、PDF和参数来源默认写入artifacts/evaluation_terrain/，文件名见脚本输出。
它们展示生成几何，不是策略通过性能或仿真画面截图。

6. 旧region布局（非本次七类标准测试）
仍可用--layout region显式选择，并指定--size-m，例如：
  bash scripts/evaluate_mine.sh --layout region --size-m 4 --dry-run
此模式从初始化到基座终点记录同样的基础数据，不采用轮入/轮出窗口。
它的起终点平台位于区域内部，保留flat/rough/slope/stairs_up/stairs_down
五类旧几何以兼容历史用途，公开入口也不接受离散障碍块。
当前6.2×6.2m七类测试请使用默认patch布局，不要加--layout region。
