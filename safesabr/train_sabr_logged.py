# python train_sabr.py 1 0 4 100_000
# param: is_dagger_train (1/0), is_obs_norm (1/0),
# parallel_env (int), ppo_step (int)

# tensorboard:
# tensorboard --logdir=./experiment_logs/starlink_high --host=0.0.0.0 --port=6006

import argparse
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize

from config import (
    DATASET_NAME,
    LOG_FILE_DIR,
    REBUF_PENALTY,
    TEST_TRACES,
    TRAIN_TRACES,
    VIDEO_BIT_RATE,
    VIDEO_SIZE_FILE,
)
from rl.dagger import DaggerTrainer
from sim_env.risk_reward_wrapper import RiskAwareRewardWrapper
from sim_env.vec_env import create_vec_env
from utils_tool import eval_func, experiment_logger, utils


class RiskMetricCallback(BaseCallback):
    def _on_step(self):
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        self._record_info_mean(infos, "rebuffer", "risk/rebuffer_mean")
        self._record_info_mean(infos, "risk_penalty", "risk/penalty_mean")
        self._record_info_mean(infos, "risk_var_rebuf", "risk/var_rebuffer")
        self._record_info_mean(infos, "risk_cvar_rebuf", "risk/cvar_rebuffer")

        penalties = self._finite_info_values(infos, "risk_penalty")
        if penalties:
            self.logger.record(
                "risk/penalty_positive_ratio",
                float(np.mean([value > 0.0 for value in penalties])),
            )
        rebufs = self._finite_info_values(infos, "rebuffer")
        if rebufs:
            self.logger.record(
                "risk/rebuffer_positive_ratio",
                float(np.mean([value > 0.0 for value in rebufs])),
            )
        return True

    def _record_info_mean(self, infos, key, metric_name):
        values = self._finite_info_values(infos, key)
        if values:
            self.logger.record(metric_name, float(np.mean(values)))

    @staticmethod
    def _finite_info_values(infos, key):
        values = []
        for info in infos:
            if key not in info:
                continue
            try:
                value = float(info[key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        return values


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SafeSABR with structured experiment logging."
    )
    parser.add_argument("is_dagger_train", type=int, help="1 to train DAgger/BC, 0 to collect only")
    parser.add_argument("is_obs_norm", type=int, help="1 to normalize observations, 0 otherwise")
    parser.add_argument("parallel_env_num", type=int, help="number of parallel training envs")
    parser.add_argument("ppo_train_step", type=int, help="PPO training steps, e.g. 100_000")
    parser.add_argument("--run-name", default=None, help="optional readable run name")
    parser.add_argument("--log-root", default="./experiment_logs", help="structured log root")
    parser.add_argument("--dagger-iters", type=int, default=15)
    parser.add_argument("--dagger-steps", type=int, default=2000)
    parser.add_argument("--dagger-epochs", type=int, default=5)
    parser.add_argument("--dagger-batch-size", type=int, default=128)
    parser.add_argument("--skip-eval", action="store_true", help="skip dagger/ppo trace evaluation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expert-algo", default="bs", choices=["bs", "mpc"], help="DAgger expert used by ABR env")
    parser.add_argument("--risk-mode", default="none", choices=["none", "mean_rebuf", "cvar_rebuf"])
    parser.add_argument("--risk-alpha", type=float, default=0.9)
    parser.add_argument("--risk-lambda", type=float, default=0.0)
    parser.add_argument("--risk-budget", type=float, default=0.0)
    parser.add_argument("--risk-window", type=int, default=512)
    return parser.parse_args()


def bool_from_int(value, name):
    if value not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1, got {value}")
    return bool(value)


def make_run_id(args, now):
    if args.run_name:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in args.run_name)
        return f"{now}_{safe}"
    return (
        f"{now}_dagger{args.is_dagger_train}_obs{args.is_obs_norm}"
        f"_env{args.parallel_env_num}_ppo{args.ppo_train_step}"
    )


def append_legacy_result(output):
    with open("Results_SABR.txt", "a") as f:
        f.write(output + "\n")


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed)


def format_eval_output(stage, result):
    return f"{stage} reward: {','.join(f'{x[0]:.3f}' for x in result)}"


def main():
    args = parse_args()
    start_time = time.time()

    is_dagger_train = bool_from_int(args.is_dagger_train, "is_dagger_train")
    is_obs_norm = bool_from_int(args.is_obs_norm, "is_obs_norm")
    parallel_env_num = args.parallel_env_num
    ppo_train_step = args.ppo_train_step

    print(f"Param | Dagger: {is_dagger_train}, obs_norm: {is_obs_norm}")
    print(f"Param | parallel env num: {parallel_env_num}, ppo_step: {ppo_train_step}")
    print(f"Param | seed: {args.seed}")
    print(
        "Param | expert: "
        f"{args.expert_algo}, risk_mode: {args.risk_mode}, "
        f"alpha: {args.risk_alpha}, lambda: {args.risk_lambda}, "
        f"budget: {args.risk_budget}, window: {args.risk_window}"
    )

    assert ppo_train_step >= 1000
    assert parallel_env_num >= 1
    assert args.dagger_iters >= 0
    assert args.dagger_steps >= 1

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = make_run_id(args, now)
    run_dir = os.path.join(args.log_root, DATASET_NAME, run_id)
    model_save_dir = os.path.join(run_dir, "models")
    tb_log_dir = os.path.join(run_dir, "tensorboard")
    event_log = os.path.join(run_dir, "events.jsonl")
    eval_log_root = os.path.join(run_dir, "raw_eval_logs")

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    run_config = {
        "command": " ".join(sys.argv),
        "dataset": DATASET_NAME,
        "train_traces": TRAIN_TRACES,
        "test_traces": TEST_TRACES,
        "trace_counts_train": experiment_logger.count_trace_files(TRAIN_TRACES),
        "trace_counts_test": experiment_logger.count_trace_files(TEST_TRACES),
        "video_bit_rate": VIDEO_BIT_RATE,
        "video_size_file": VIDEO_SIZE_FILE,
        "rebuf_penalty": REBUF_PENALTY,
        "legacy_log_file_dir": LOG_FILE_DIR,
        "args": vars(args),
        "run_dir": run_dir,
        "model_save_dir": model_save_dir,
        "tensorboard_dir": tb_log_dir,
    }
    experiment_logger.write_run_config(run_dir, run_config, code_root=".")
    experiment_logger.append_jsonl(event_log, {"event": "run_start", "run_id": run_id})

    print(f"Run directory: {run_dir}")
    print(f"TensorBoard log directory: {tb_log_dir}")
    print(f"Models and VecNormalize stats will be saved in: {model_save_dir}")

    set_global_seed(args.seed)

    vec_env = create_vec_env(
        TRAIN_TRACES,
        parallel_env_num,
        seed=args.seed,
        expert_algo=args.expert_algo,
    )
    train_env = RiskAwareRewardWrapper(
        vec_env,
        mode="none",
        alpha=args.risk_alpha,
        penalty_lambda=0.0,
        budget=args.risk_budget,
        window=args.risk_window,
    )
    norm_env = VecNormalize(train_env, norm_obs=is_obs_norm, norm_reward=True, clip_reward=10.0)

    model = PPO(
        policy="MlpPolicy",
        env=norm_env,
        verbose=1,
        device=args.device,
        tensorboard_log=tb_log_dir,
        n_steps=2048 // parallel_env_num,
        seed=args.seed,
    )

    tb_log_file = os.path.join(tb_log_dir, "sb3")
    tb_logger = configure(tb_log_file, ["stdout", "tensorboard"])
    model.set_logger(tb_logger)

    rl_model_save_dir = os.path.join(model_save_dir, "rl_model")
    dagger_model_dir = os.path.join(rl_model_save_dir, "dagger")
    ppo_model_dir = os.path.join(rl_model_save_dir, "ppo")

    print(f"dagger iteration: {args.dagger_iters}")
    dagger_trainer = DaggerTrainer(model, norm_env, logger=model.logger)

    experiment_logger.append_jsonl(
        event_log,
        {
            "event": "dagger_start",
            "enabled": is_dagger_train,
            "iters": args.dagger_iters,
            "steps_per_iter": args.dagger_steps,
            "epochs_per_iter": args.dagger_epochs,
            "batch_size": args.dagger_batch_size,
        },
    )
    if is_dagger_train:
        dagger_trainer.run(
            dagger_iters=args.dagger_iters,
            steps_per_iter=args.dagger_steps,
            epochs_per_iter=args.dagger_epochs,
            batch_size=args.dagger_batch_size,
        )
    else:
        dagger_trainer.run_notrain(
            dagger_iters=args.dagger_iters,
            steps_per_iter=args.dagger_steps,
            epochs_per_iter=args.dagger_epochs,
            batch_size=args.dagger_batch_size,
        )

    utils.save_env_and_model(model, norm_env, dagger_model_dir)
    experiment_logger.append_jsonl(
        event_log,
        {"event": "dagger_finish", "model_dir": dagger_model_dir},
    )

    if args.skip_eval:
        dagger_test_result = []
        dagger_reward_output = "Dagger reward: skipped"
    else:
        dagger_test_result, dagger_details = eval_func.eval_model_trace_list(
            dagger_model_dir,
            os.path.join(eval_log_root, "dagger"),
            TEST_TRACES,
            "ppo_sb",
            "test_ppo_sb.py",
            return_details=True,
        )
        experiment_logger.write_eval_outputs(run_dir, "dagger", dagger_details)
        dagger_reward_output = format_eval_output("Dagger", dagger_test_result)

    print(dagger_reward_output)
    experiment_logger.append_jsonl(
        event_log,
        {"event": "dagger_eval_finish", "result": dagger_test_result},
    )

    experiment_logger.append_jsonl(
        event_log,
        {
            "event": "ppo_start",
            "total_timesteps": ppo_train_step,
            "risk_mode": args.risk_mode,
            "risk_alpha": args.risk_alpha,
            "risk_lambda": args.risk_lambda,
            "risk_budget": args.risk_budget,
            "risk_window": args.risk_window,
        },
    )
    train_env.set_risk_config(
        mode=args.risk_mode,
        alpha=args.risk_alpha,
        penalty_lambda=args.risk_lambda,
        budget=args.risk_budget,
        window=args.risk_window,
        reset_history=True,
    )
    model.learn(
        total_timesteps=ppo_train_step,
        tb_log_name="ppo",
        callback=RiskMetricCallback(),
    )
    risk_train_state = train_env.get_risk_state()
    utils.save_env_and_model(model, norm_env, ppo_model_dir)
    experiment_logger.append_jsonl(
        event_log,
        {
            "event": "ppo_finish",
            "model_dir": ppo_model_dir,
            "risk_train_state": risk_train_state,
        },
    )

    if args.skip_eval:
        ppo_test_result = []
        ppo_reward_output = "PPO reward: skipped"
    else:
        ppo_test_result, ppo_details = eval_func.eval_model_trace_list(
            ppo_model_dir,
            os.path.join(eval_log_root, "ppo"),
            TEST_TRACES,
            "ppo_sb",
            "test_ppo_sb.py",
            return_details=True,
        )
        experiment_logger.write_eval_outputs(run_dir, "ppo", ppo_details)
        ppo_reward_output = format_eval_output("PPO", ppo_test_result)

    elapsed_s = time.time() - start_time
    print("Training finished.")

    output = (
        f"Run: {run_id} | "
        f"Train dagger: {is_dagger_train}, ppo_step: {ppo_train_step} | "
        f"Obs_norm: {is_obs_norm} | "
        f"env num: {parallel_env_num} | "
        f"{dagger_reward_output} | "
        f"{ppo_reward_output} | "
        f"elapsed_s: {elapsed_s:.1f}"
    )

    print(output)
    append_legacy_result(output)
    experiment_logger.write_json(
        os.path.join(run_dir, "final_summary.json"),
        {
            "run_id": run_id,
            "dataset": DATASET_NAME,
            "dagger_result": dagger_test_result,
            "ppo_result": ppo_test_result,
            "risk_train_state": risk_train_state,
            "elapsed_s": elapsed_s,
            "run_dir": run_dir,
        },
    )
    experiment_logger.append_jsonl(
        event_log,
        {
            "event": "run_finish",
            "elapsed_s": elapsed_s,
            "dagger_result": dagger_test_result,
            "ppo_result": ppo_test_result,
            "risk_train_state": risk_train_state,
        },
    )
    print("==============================")


if __name__ == "__main__":
    main()
