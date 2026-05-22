#!/usr/bin/env bash
set -euo pipefail

python tools/make_synthetic_video_size.py
python tools/prepare_starlink_traces.py --source-root data/starnet_pkl

cd safesabr
bash build_env_c_plus/build_rl.sh

python train_sabr.py 1 0 4 100000 \
  --seed 42 \
  --risk-mode cvar_rebuf \
  --risk-alpha 0.95 \
  --risk-lambda 20 \
  --run-name safesabr_seed42

