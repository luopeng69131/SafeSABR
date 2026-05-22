import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from statistics import mean


def _json_default(obj):
    try:
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return str(obj)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path, payload):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")


def append_jsonl(path, payload):
    ensure_dir(os.path.dirname(path))
    payload = {"time": datetime.now().isoformat(timespec="seconds"), **payload}
    with open(path, "a") as f:
        f.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")


def command_output(cmd, cwd=None):
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def package_versions():
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for module_name in ["stable_baselines3", "torch", "numpy", "pandas", "gymnasium"]:
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[module_name] = None
    return versions


def count_trace_files(paths):
    if paths is None:
        return {}
    if isinstance(paths, str):
        paths = [paths]
    counts = {}
    for path in paths:
        if os.path.isdir(path):
            counts[path] = len(
                [
                    name
                    for name in os.listdir(path)
                    if os.path.isfile(os.path.join(path, name))
                ]
            )
        else:
            counts[path] = None
    return counts


def write_run_config(run_dir, payload, code_root="."):
    payload = dict(payload)
    payload["created_at"] = datetime.now().isoformat(timespec="seconds")
    payload["versions"] = package_versions()
    payload["git_commit"] = command_output(["git", "rev-parse", "--short", "HEAD"], cwd=code_root)
    payload["git_status_short"] = command_output(["git", "status", "--short"], cwd=code_root)
    write_json(os.path.join(run_dir, "run_config.json"), payload)


def aggregate_trace_metrics(rows):
    numeric_keys = [
        "reward_sum_excl_first",
        "reward_sum",
        "mean_reward",
        "mean_entropy",
        "mean_buffer",
        "mean_bitrate",
        "total_rebuffer",
        "mean_rebuffer",
        "p95_rebuffer",
        "p99_rebuffer",
        "cvar90_rebuffer",
        "cvar95_rebuffer",
        "max_rebuffer",
        "rebuffer_ratio",
        "rebuffer_ratio_over_0_5",
        "rebuffer_ratio_over_1_0",
        "switch_count",
        "switch_rate",
        "mean_delay_ms",
        "shield_rate",
        "safe_cap_kbps_mean",
        "chunks",
    ]
    aggregate = {"num_traces": len(rows)}
    for key in numeric_keys:
        vals = [row.get(key) for row in rows if row.get(key) is not None]
        aggregate[key] = mean(vals) if vals else None
    return aggregate


def flatten_eval_details(eval_details):
    rows = []
    dataset_summaries = []
    for dataset in eval_details:
        dataset_id = dataset.get("dataset_id")
        dataset_path = dataset.get("trace_dataset")
        summary = dataset.get("summary", {})
        trace_rows = dataset.get("traces", [])
        dataset_summaries.append(
            {
                "dataset_id": dataset_id,
                "trace_dataset": dataset_path,
                **summary,
            }
        )
        for row in trace_rows:
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "trace_dataset": dataset_path,
                    **row,
                }
            )
    return rows, dataset_summaries


def write_csv(path, rows):
    ensure_dir(os.path.dirname(path))
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_eval_outputs(run_dir, stage, eval_details):
    eval_dir = ensure_dir(os.path.join(run_dir, "eval_summary"))
    trace_rows, dataset_rows = flatten_eval_details(eval_details)
    aggregate = aggregate_trace_metrics(trace_rows)
    payload = {
        "stage": stage,
        "aggregate_trace_mean": aggregate,
        "datasets": eval_details,
    }
    write_json(os.path.join(eval_dir, f"{stage}.json"), payload)
    write_csv(os.path.join(eval_dir, f"{stage}_traces.csv"), trace_rows)
    write_csv(os.path.join(eval_dir, f"{stage}_datasets.csv"), dataset_rows)
    return payload
