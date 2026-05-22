import os
import shutil
import subprocess
import sys

import numpy as np


def _safe_mean(values):
    values = [v for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def _safe_percentile(values, q):
    values = [v for v in values if np.isfinite(v)]
    return float(np.percentile(values, q)) if values else float("nan")


def _safe_cvar(values, alpha):
    values = sorted((v for v in values if np.isfinite(v)), reverse=True)
    if not values:
        return float("nan")
    tail_count = max(1, int(np.ceil((1.0 - alpha) * len(values))))
    return float(np.mean(values[:tail_count]))


def parse_ppo_log(log_path):
    rows = []
    with open(log_path, "r") as f:
        for line in f:
            parse = line.split()
            if len(parse) < 7:
                continue
            try:
                if len(parse) >= 8:
                    entropy = float(parse[6])
                    reward = float(parse[7])
                else:
                    entropy = float("nan")
                    reward = float(parse[6])
                rows.append(
                    {
                        "time_s": float(parse[0]),
                        "bitrate_kbps": float(parse[1]),
                        "buffer_s": float(parse[2]),
                        "rebuffer_s": float(parse[3]),
                        "chunk_size_bytes": float(parse[4]),
                        "delay_ms": float(parse[5]),
                        "entropy": entropy,
                        "reward": reward,
                        "raw_action": float(parse[8]) if len(parse) > 8 else float("nan"),
                        "safe_cap_kbps": float(parse[9]) if len(parse) > 9 else float("nan"),
                        "shielded_action": float(parse[10]) if len(parse) > 10 else float("nan"),
                    }
                )
            except ValueError:
                continue

    if not rows:
        return {
            "trace_file": os.path.basename(log_path),
            "chunks": 0,
            "reward_sum_excl_first": float("nan"),
            "reward_sum": float("nan"),
            "mean_reward": float("nan"),
            "mean_entropy": float("nan"),
            "mean_buffer": float("nan"),
            "mean_bitrate": float("nan"),
            "total_rebuffer": float("nan"),
            "mean_rebuffer": float("nan"),
            "p95_rebuffer": float("nan"),
            "p99_rebuffer": float("nan"),
            "cvar90_rebuffer": float("nan"),
            "cvar95_rebuffer": float("nan"),
            "max_rebuffer": float("nan"),
            "rebuffer_ratio": float("nan"),
            "rebuffer_ratio_over_0_5": float("nan"),
            "rebuffer_ratio_over_1_0": float("nan"),
            "switch_count": float("nan"),
            "switch_rate": float("nan"),
            "mean_delay_ms": float("nan"),
        }

    rewards = [row["reward"] for row in rows]
    entropies = [row["entropy"] for row in rows]
    buffers = [row["buffer_s"] for row in rows]
    bitrates = [row["bitrate_kbps"] for row in rows]
    rebufs = [row["rebuffer_s"] for row in rows]
    delays = [row["delay_ms"] for row in rows]
    shielded = [row["shielded_action"] for row in rows]
    safe_caps = [row["safe_cap_kbps"] for row in rows]
    switch_count = sum(
        1 for prev, curr in zip(bitrates[:-1], bitrates[1:]) if curr != prev
    )
    chunks = len(rows)

    return {
        "trace_file": os.path.basename(log_path),
        "chunks": chunks,
        "reward_sum_excl_first": float(np.sum(rewards[1:])),
        "reward_sum": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)),
        "mean_entropy": _safe_mean(entropies[1:]),
        "mean_buffer": _safe_mean(buffers[1:]),
        "mean_bitrate": _safe_mean(bitrates),
        "total_rebuffer": float(np.sum(rebufs)),
        "mean_rebuffer": _safe_mean(rebufs),
        "p95_rebuffer": _safe_percentile(rebufs, 95),
        "p99_rebuffer": _safe_percentile(rebufs, 99),
        "cvar90_rebuffer": _safe_cvar(rebufs, 0.90),
        "cvar95_rebuffer": _safe_cvar(rebufs, 0.95),
        "max_rebuffer": float(np.max(rebufs)),
        "rebuffer_ratio": float(np.mean([x > 0 for x in rebufs])),
        "rebuffer_ratio_over_0_5": float(np.mean([x > 0.5 for x in rebufs])),
        "rebuffer_ratio_over_1_0": float(np.mean([x > 1.0 for x in rebufs])),
        "switch_count": switch_count,
        "switch_rate": float(switch_count / max(chunks - 1, 1)),
        "mean_delay_ms": _safe_mean(delays),
        "shield_rate": _safe_mean(shielded),
        "safe_cap_kbps_mean": _safe_mean(safe_caps),
    }


def summarize_log_dir(test_log_dir, file_name="ppo_sb"):
    traces = []
    for test_log_file in sorted(os.listdir(test_log_dir)):
        if file_name not in test_log_file:
            continue
        traces.append(parse_ppo_log(os.path.join(test_log_dir, test_log_file)))

    rewards = [trace["reward_sum_excl_first"] for trace in traces]
    entropies = [trace["mean_entropy"] for trace in traces]
    buffers = [trace["mean_buffer"] for trace in traces]

    summary = {
        "num_traces": len(traces),
        "reward_mean": _safe_mean(rewards),
        "entropy_mean": _safe_mean(entropies),
        "buffer_mean": _safe_mean(buffers),
        "reward_median": _safe_percentile(rewards, 50),
        "reward_p05": _safe_percentile(rewards, 5),
        "reward_p95": _safe_percentile(rewards, 95),
        "total_rebuffer_mean": _safe_mean([trace["total_rebuffer"] for trace in traces]),
        "p95_rebuffer_mean": _safe_mean([trace["p95_rebuffer"] for trace in traces]),
        "p99_rebuffer_mean": _safe_mean([trace["p99_rebuffer"] for trace in traces]),
        "cvar90_rebuffer_mean": _safe_mean([trace["cvar90_rebuffer"] for trace in traces]),
        "cvar95_rebuffer_mean": _safe_mean([trace["cvar95_rebuffer"] for trace in traces]),
        "max_rebuffer_mean": _safe_mean([trace["max_rebuffer"] for trace in traces]),
        "rebuffer_ratio_mean": _safe_mean([trace["rebuffer_ratio"] for trace in traces]),
        "rebuffer_ratio_over_0_5_mean": _safe_mean(
            [trace["rebuffer_ratio_over_0_5"] for trace in traces]
        ),
        "rebuffer_ratio_over_1_0_mean": _safe_mean(
            [trace["rebuffer_ratio_over_1_0"] for trace in traces]
        ),
        "mean_bitrate_mean": _safe_mean([trace["mean_bitrate"] for trace in traces]),
        "switch_rate_mean": _safe_mean([trace["switch_rate"] for trace in traces]),
        "shield_rate_mean": _safe_mean([trace["shield_rate"] for trace in traces]),
        "safe_cap_kbps_mean": _safe_mean([trace["safe_cap_kbps_mean"] for trace in traces]),
    }
    return summary, traces


def eval_model(
    model_path,
    test_trace="",
    test_log_dir="",
    file_name="ppo_sb",
    script_name="test_ppo_sb.py",
    extra_args=None,
    return_details=False,
):
    # use for eval_model_list
    # filter log (file_name) and set test script (script_name)
    cmd = [sys.executable, script_name, model_path, test_trace, test_log_dir]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(
        cmd,
        check=True,
    )
    summary, traces = summarize_log_dir(test_log_dir, file_name)
    result_tuple = (
        summary["reward_mean"],
        summary["entropy_mean"],
        summary["buffer_mean"],
    )
    if return_details:
        return result_tuple, summary, traces
    return result_tuple


def eval_model_trace_list(
    model_path,
    test_log_dir="./test_results/",
    test_trace_datasets="",
    file_name="ppo_sb",
    script_name="test_ppo_sb.py",
    extra_args=None,
    return_details=False,
):
    # use for eval_model_list
    # test_trace_datasets: list or str for paths
    if not isinstance(test_trace_datasets, list):
        test_trace_datasets = [test_trace_datasets]

    result = []
    details = []
    for i, trace_dataset in enumerate(test_trace_datasets):
        sub_log_dir = os.path.join(test_log_dir, f"trace_log{i+1}")
        # if not clear, when change dataset, previous logs will corrupt metrics.
        if os.path.exists(sub_log_dir):
            shutil.rmtree(sub_log_dir)

        os.makedirs(sub_log_dir, exist_ok=True)

        test_result = eval_model(
            model_path,
            trace_dataset,
            sub_log_dir,
            file_name,
            script_name,
            extra_args=extra_args,
            return_details=return_details,
        )
        if return_details:
            test_result, summary, traces = test_result
            details.append(
                {
                    "dataset_id": i + 1,
                    "trace_dataset": trace_dataset,
                    "log_dir": sub_log_dir,
                    "summary": summary,
                    "traces": traces,
                }
            )
        result.append(test_result)

    if return_details:
        return result, details
    return result
