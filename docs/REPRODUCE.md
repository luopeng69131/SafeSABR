# Reproduction Workflow

This repository provides the SafeSABR method code and trace-preparation scripts. It intentionally excludes StarNet data, large logs, trained checkpoints, and third-party baselines.

## 1. Environment

```bash
conda env create -f environment.yml
conda activate safesabr
```

## 2. Prepare ABR Inputs

Generate synthetic high-bitrate chunk-size files:

```bash
python tools/make_synthetic_video_size.py
```

Convert StarNet processed throughput data into SABR replay traces:

```bash
python tools/prepare_starlink_traces.py --source-root data/starnet_pkl
```

## 3. Build the Expert Backend

Behavior-cloning pretraining uses a beam-search expert implemented in C++:

```bash
cd safesabr
bash build_env_c_plus/build_rl.sh
```

## 4. Train SafeSABR

From the `safesabr/` directory:

```bash
python train_sabr.py 1 0 4 100000 \
  --seed 42 \
  --risk-mode cvar_rebuf \
  --risk-alpha 0.95 \
  --risk-lambda 20 \
  --run-name safesabr_seed42
```

The positional arguments are:

```text
is_dagger_train is_obs_norm parallel_env_num ppo_train_step
```

The example above enables behavior-cloning pretraining, disables observation normalization, uses four parallel environments, and runs 100,000 PPO steps.

## 5. Safe-Capacity Inputs

The runtime auditor can use a generated safe-capacity CSV through `--shield external_predictor`. The expected CSV fields include:

```text
trace_id, split, time_s, safe_cap_kbps, predictor
```

If you have StarNet/BG-CFQS predictor outputs, use:

```bash
python tools/export_predictor_safe_caps.py \
  --summary-dir data/starnet_model_summary \
  --source-root data/starnet_pkl
```

## 6. Evaluate Runtime Auditing

From the `safesabr/` directory:

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

Evaluation logs are written under `safesabr/experiment_logs/`, which is ignored by git.
