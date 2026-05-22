#!/usr/bin/env bash
set -euo pipefail

cd safesabr

MODEL_DIR="${MODEL_DIR:-}"
if [[ -z "$MODEL_DIR" ]]; then
  echo "Set MODEL_DIR to a trained SafeSABR model directory, for example:"
  echo "MODEL_DIR=./experiment_logs/starlink_high/<run-id>/models/rl_model/ppo bash ../scripts/run_auditor_eval_example.sh"
  exit 1
fi

python evaluate_action_shield.py \
  --log-root ./experiment_logs/starlink_high \
  --model-dir "$MODEL_DIR" \
  --model-label safesabr \
  --shield external_predictor \
  --external-predictor-file ./experiment_logs/starlink_high/predictor_safe_caps/BG-CFQS_safe_caps.csv \
  --external-predictor-name BG-CFQS \
  --shield-margin 0.90 \
  --shield-low-buffer-margin 0.90
