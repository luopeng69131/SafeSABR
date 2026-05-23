# SafeSABR

<p align="right">
  <a href="README.md">English</a>
</p>

**面向 Starlink 网络的风险校准自适应码率流媒体传输。**

SafeSABR 是一个面向 Starlink 高码率视频流的学习型自适应码率（Adaptive Bitrate, ABR）框架。它关注一个仅靠平均 QoE 很容易被忽略的问题：在 Starlink 切换和吞吐骤降期间，学习型 ABR 策略可能仍然持续请求激进的高码率视频块，进而导致严重的会话级卡顿。

> 论文：arXiv 链接后续更新。

<p align="center">
  <img src="assets/safesabr_overview.png" alt="SafeSABR overview" width="88%">
</p>

## 为什么需要 SafeSABR？

Starlink 让缺少地面宽带覆盖区域的高码率视频传输成为可能，但其接入吞吐会受到卫星移动和切换等因素影响而快速波动。这使 ABR 决策不再只是追求更高平均 QoE，而是需要同时考虑 QoE 和严重卡顿风险：激进码率选择在链路良好时能提升画质，但在突然掉速时，少数错误决策就可能耗尽播放缓存并造成长时间卡顿。

<p align="center">
  <img src="assets/starlink_rebuffer_challenge.png" alt="Starlink ABR rebuffering challenge" width="82%">
</p>

SafeSABR 采用三阶段设计：

1. **行为克隆预训练**：从专家策略中学习高 QoE 的 ABR 初始策略。
2. **风险校准 RL 微调**：通过严重卡顿尾部惩罚，使策略减少高风险动作倾向。
3. **运行时安全审计**：在执行前根据 safe-capacity 估计检查策略请求的码率，并修正潜在危险动作。

<p align="center">
  <img src="assets/safesabr_framework.png" alt="SafeSABR framework" width="92%">
</p>

## 仓库内容

本仓库提供 SafeSABR 的核心实现：

- SafeSABR 训练代码，包括行为克隆预训练和风险校准 PPO 微调。
- 基于 safe-capacity 的运行时安全审计代码。
- StarNet 到 SABR 回放 trace 的转换工具。
- 面向 4K/8K 风格 ABR 实验的高码率视频块大小生成工具。
- 用于说明问题背景和方法设计的论文图。

## 仓库结构

```text
SafeSABR/
├── assets/                    # README 和项目主页使用的论文风格图片
├── docs/                      # 数据准备和复现实验说明
├── safesabr/                  # SafeSABR 核心代码
│   ├── train_sabr.py          # BC + RL 微调入口
│   ├── train_sabr_logged.py   # 带结构化日志的训练实现
│   ├── evaluate_action_shield.py
│   ├── test_ppo_sb.py         # 带运行时安全审计的评估流程
│   ├── sim_env/               # ABR 仿真环境
│   ├── rl/                    # 行为克隆 / DAgger 工具
│   ├── utils_tool/            # 评估和日志工具
│   └── build_env_c_plus/      # 行为克隆专家策略的 C++ 后端
└── tools/
    ├── prepare_starlink_traces.py
    ├── make_synthetic_video_size.py
    └── export_predictor_safe_caps.py
```

## 数据

SafeSABR 使用由 StarNet 数据集处理得到的 Starlink 测量 trace：

https://github.com/ConnectedSystemsLab/StarNet

请先根据 StarNet 官方仓库获取或准备处理后的吞吐数据，然后参考 [docs/DATA.md](docs/DATA.md) 将其转换为 SABR 可回放的 trace 格式。

## 快速开始

创建环境：

```bash
conda env create -f environment.yml
conda activate safesabr
```

生成高码率视频块大小文件：

```bash
python tools/make_synthetic_video_size.py
```

将处理后的 StarNet 数据放到 `data/starnet_pkl` 后，生成 SABR trace：

```bash
python tools/prepare_starlink_traces.py --source-root data/starnet_pkl
```

编译行为克隆预训练使用的 C++ 专家后端：

```bash
cd safesabr
bash build_env_c_plus/build_rl.sh
```

训练 SafeSABR 策略：

```bash
python train_sabr.py 1 0 4 100000 \
  --seed 42 \
  --risk-mode cvar_rebuf \
  --risk-alpha 0.95 \
  --risk-lambda 20 \
  --run-name safesabr_seed42
```

使用运行时安全审计进行评估：

```bash
python evaluate_action_shield.py \
  --log-root ./experiment_logs/starlink_high \
  --model-dir ./experiment_logs/starlink_high/<run-id>/models/rl_model/ppo \
  --model-label safesabr \
  --shield external_predictor \
  --external-predictor-file ./experiment_logs/starlink_high/predictor_safe_caps/BG-CFQS_safe_caps.csv \
  --external-predictor-name BG-CFQS \
  --shield-margin 0.90 \
  --shield-low-buffer-margin 0.90
```

完整流程可参考 [docs/REPRODUCE.md](docs/REPRODUCE.md)。

## 结果示意

论文中使用 QoE--severe-risk operating point 来评价 SafeSABR，而不是只看平均 QoE。

<p align="center">
  <img src="assets/qoe_severe_risk_tradeoff.png" alt="QoE severe-risk tradeoff" width="86%">
</p>

## 致谢

SafeSABR 受益于 ABR 方向的开源研究生态。感谢 [Comyco-Lin](https://github.com/godka/comyco-lin) 和 [Pensieve Retrain](https://github.com/GreenLv/pensieve_retrain) 作者公开的实现，这些项目为 ABR 仿真、训练和评估流程提供了重要参考。
