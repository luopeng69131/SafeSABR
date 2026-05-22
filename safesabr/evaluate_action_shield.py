#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import re
from datetime import datetime

from config import DATASET_NAME, FINE_TUNE_TRACES, TEST_TRACES, TRAIN_TRACES
from utils_tool import eval_func, experiment_logger


def parse_int_list(value):
    return [int(item) for item in value.split(",") if item.strip()]


def select_trace_set(name):
    if name == "test":
        return TEST_TRACES
    if name == "calib":
        return FINE_TUNE_TRACES
    if name == "train":
        return TRAIN_TRACES
    raise ValueError(f"Unsupported trace set: {name}")


def find_run(log_root, seed, ppo_k):
    pattern = os.path.join(log_root, f"20*starlink_high_obs0_ppo{ppo_k}k_seed{seed}")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No run matched {pattern}")
    return matches[-1]


def weighted_summary(eval_details):
    rows = []
    def as_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    for dataset in eval_details:
        summary = dataset["summary"]
        rows.append(
            {
                "num_traces": int(summary["num_traces"]),
                "reward_mean": float(summary["reward_mean"]),
                "total_rebuffer_mean": float(summary["total_rebuffer_mean"]),
                "p95_rebuffer_mean": float(summary["p95_rebuffer_mean"]),
                "p99_rebuffer_mean": as_float(summary.get("p99_rebuffer_mean")),
                "cvar90_rebuffer_mean": as_float(summary.get("cvar90_rebuffer_mean")),
                "cvar95_rebuffer_mean": as_float(summary.get("cvar95_rebuffer_mean")),
                "max_rebuffer_mean": as_float(summary.get("max_rebuffer_mean")),
                "mean_bitrate_mean": float(summary["mean_bitrate_mean"]),
                "switch_rate_mean": float(summary["switch_rate_mean"]),
                "buffer_mean": float(summary["buffer_mean"]),
                "shield_rate_mean": as_float(summary.get("shield_rate_mean")),
                "safe_cap_kbps_mean": as_float(summary.get("safe_cap_kbps_mean")),
            }
        )
    total = sum(row["num_traces"] for row in rows)

    def wavg(key):
        return sum(row[key] * row["num_traces"] for row in rows) / total

    return {
        "num_traces": total,
        "reward_mean": wavg("reward_mean"),
        "total_rebuffer_mean": wavg("total_rebuffer_mean"),
        "p95_rebuffer_mean": wavg("p95_rebuffer_mean"),
        "p99_rebuffer_mean": wavg("p99_rebuffer_mean"),
        "cvar90_rebuffer_mean": wavg("cvar90_rebuffer_mean"),
        "cvar95_rebuffer_mean": wavg("cvar95_rebuffer_mean"),
        "max_rebuffer_mean": wavg("max_rebuffer_mean"),
        "mean_bitrate_mean": wavg("mean_bitrate_mean"),
        "switch_rate_mean": wavg("switch_rate_mean"),
        "buffer_mean": wavg("buffer_mean"),
        "shield_rate_mean": wavg("shield_rate_mean"),
        "safe_cap_kbps_mean": wavg("safe_cap_kbps_mean"),
    }


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained SafeSABR models with runtime safety auditing.")
    parser.add_argument("--log-root", default="./experiment_logs/starlink_high")
    parser.add_argument("--seeds", default="7,42,2024")
    parser.add_argument("--ppo-ks", default="30,50")
    parser.add_argument("--shield", default="lower_quantile")
    parser.add_argument("--shield-history", type=int, default=5)
    parser.add_argument("--shield-quantile", type=float, default=0.2)
    parser.add_argument("--shield-margin", type=float, default=0.75)
    parser.add_argument("--shield-low-buffer-threshold", type=float, default=10.0)
    parser.add_argument("--shield-low-buffer-margin", type=float, default=0.65)
    parser.add_argument("--shield-buffer-guard", type=float, default=1.0)
    parser.add_argument("--predictor-history", type=int, default=15)
    parser.add_argument("--predictor-error-window", type=int, default=5)
    parser.add_argument("--predictor-trend-window", type=int, default=5)
    parser.add_argument("--predictor-volatility-z", type=float, default=0.15)
    parser.add_argument("--predictor-trend-gain", type=float, default=1.0)
    parser.add_argument("--external-predictor-file", default="")
    parser.add_argument("--external-predictor-name", default="")
    parser.add_argument("--external-predictor-fallback", default="history_quantile")
    parser.add_argument("--external-predictor-max-age", type=float, default=120.0)
    parser.add_argument("--trace-set", choices=["test", "calib", "train"], default="test")
    parser.add_argument("--out-name", default=None)
    parser.add_argument(
        "--model-dir",
        default="",
        help=(
            "Directly evaluate one trained model directory, e.g. "
            "./experiment_logs/starlink_high/<run>/models/rl_model/ppo. "
            "When this is set, --seeds and --ppo-ks are ignored."
        ),
    )
    parser.add_argument("--model-label", default="safesabr")
    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    ppo_ks = parse_int_list(args.ppo_ks)
    trace_datasets = select_trace_set(args.trace_set)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    shield_tag = (
        f"{args.shield}_q{int(args.shield_quantile * 100):02d}"
        f"_m{int(args.shield_margin * 100):02d}"
        f"_lbm{int(args.shield_low_buffer_margin * 100):02d}"
    )
    out_name = args.out_name or f"{now}_{shield_tag}"
    out_dir = os.path.join(args.log_root, "shield_eval", out_name)
    os.makedirs(out_dir, exist_ok=True)

    extra_args = [
        "--shield",
        args.shield,
        "--shield-history",
        str(args.shield_history),
        "--shield-quantile",
        str(args.shield_quantile),
        "--shield-margin",
        str(args.shield_margin),
        "--shield-low-buffer-threshold",
        str(args.shield_low_buffer_threshold),
        "--shield-low-buffer-margin",
        str(args.shield_low_buffer_margin),
        "--shield-buffer-guard",
        str(args.shield_buffer_guard),
        "--predictor-history",
        str(args.predictor_history),
        "--predictor-error-window",
        str(args.predictor_error_window),
        "--predictor-trend-window",
        str(args.predictor_trend_window),
        "--predictor-volatility-z",
        str(args.predictor_volatility_z),
        "--predictor-trend-gain",
        str(args.predictor_trend_gain),
    ]
    if args.external_predictor_file:
        extra_args.extend(
            [
                "--external-predictor-file",
                args.external_predictor_file,
                "--external-predictor-name",
                args.external_predictor_name,
                "--external-predictor-fallback",
                args.external_predictor_fallback,
                "--external-predictor-max-age",
                str(args.external_predictor_max_age),
            ]
        )

    config = {
        "dataset": DATASET_NAME,
        "trace_set": args.trace_set,
        "test_traces": trace_datasets,
        "seeds": seeds,
        "ppo_ks": ppo_ks,
        "shield": vars(args),
        "extra_args": extra_args,
        "out_dir": out_dir,
    }
    experiment_logger.write_json(os.path.join(out_dir, "shield_config.json"), config)

    summary_rows = []

    if args.model_dir:
        label = args.model_label
        print(f"[eval] {label} from {args.model_dir}")
        _, details = eval_func.eval_model_trace_list(
            args.model_dir,
            os.path.join(out_dir, "raw_eval_logs", label),
            trace_datasets,
            "ppo_sb",
            "test_ppo_sb.py",
            extra_args=extra_args,
            return_details=True,
        )
        experiment_logger.write_eval_outputs(out_dir, label, details)
        summary_rows.append(
            {
                "label": label,
                "seed": "",
                "source_ppo_k": "",
                "stage": "direct",
                "source_run": args.model_dir,
                **weighted_summary(details),
            }
        )
        write_csv(os.path.join(out_dir, "shield_summary.csv"), summary_rows)
        print(f"[done] wrote {os.path.join(out_dir, 'shield_summary.csv')}")
        return

    evaluated_dagger_seeds = set()

    for seed in seeds:
        for ppo_k in ppo_ks:
            run_dir = find_run(args.log_root, seed, ppo_k)
            run_name = os.path.basename(run_dir)

            if seed not in evaluated_dagger_seeds:
                dagger_model_dir = os.path.join(run_dir, "models", "rl_model", "dagger")
                label = f"seed{seed}_dagger_shield"
                print(f"[eval] {label} from {run_name}")
                _, details = eval_func.eval_model_trace_list(
                    dagger_model_dir,
                    os.path.join(out_dir, "raw_eval_logs", label),
                    trace_datasets,
                    "ppo_sb",
                    "test_ppo_sb.py",
                    extra_args=extra_args,
                    return_details=True,
                )
                experiment_logger.write_eval_outputs(out_dir, label, details)
                summary_rows.append(
                    {
                        "label": label,
                        "seed": seed,
                        "source_ppo_k": ppo_k,
                        "stage": "dagger",
                        "source_run": run_name,
                        **weighted_summary(details),
                    }
                )
                evaluated_dagger_seeds.add(seed)

            ppo_model_dir = os.path.join(run_dir, "models", "rl_model", "ppo")
            label = f"seed{seed}_ppo{ppo_k}k_shield"
            print(f"[eval] {label} from {run_name}")
            _, details = eval_func.eval_model_trace_list(
                ppo_model_dir,
                os.path.join(out_dir, "raw_eval_logs", label),
                trace_datasets,
                "ppo_sb",
                "test_ppo_sb.py",
                extra_args=extra_args,
                return_details=True,
            )
            experiment_logger.write_eval_outputs(out_dir, label, details)
            summary_rows.append(
                {
                    "label": label,
                    "seed": seed,
                    "source_ppo_k": ppo_k,
                    "stage": "ppo",
                    "source_run": run_name,
                    **weighted_summary(details),
                }
            )

    write_csv(os.path.join(out_dir, "shield_summary.csv"), summary_rows)
    print(f"[done] wrote {os.path.join(out_dir, 'shield_summary.csv')}")


if __name__ == "__main__":
    main()
