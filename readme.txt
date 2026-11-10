当前训练任务：nezha2_mine（门控模态双编码器）。
训练命令：bash scripts/train_mine.sh --max_iterations 10000 --headless
五组消融：bash scripts/train_mine_ablation.sh --max_iterations 10000 --seeds 10,20,30 --headless
环境安装、配置位置和平均奖励 CSV 说明见 TRAINING_ENV.txt。
独立Isaac Gym评估：bash scripts/evaluate_mine.sh --help
成功率、功率/能量、COT、速度不稳定率及跟踪误差所需基础数据CSV说明：
wheel_legged_gym/evaluation/README.txt（默认6.2×6.2米、7类地形，首轮进入至末轮从对侧离开）。
