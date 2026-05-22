#  python test_ppo_sb.py {model_dir} {test_trace_dir} {log_file_dir}
# now: python test_ppo_sb.py ./model_data/rl_model/dagger {test_trace_dir} {log_file_dir}
#  python test_ppo_sb.py ./model_data/rl_model/ppo 
#  python test_ppo_sb.py ./model_data/rl_model/ppo default default

import os
import csv
import pickle

import sys
import numpy as np
import torch
from bisect import bisect_right

from utils_tool import utils

from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv  # 新增

from sim_env import load_trace
import sim_env.fixed_env as env
# from sim_env.abr_gym_env import ABRGymEnv

from config import VIDEO_BIT_RATE, REBUF_PENALTY, TEST_TRACES, TRAIN_TRACES, LOG_FILE_DIR

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 配置参数
S_INFO = 6  
S_LEN = 8  
A_DIM = 6

BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0

SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  
RANDOM_SEED = 42

NORMALIZED = True #True
# ---------------------------------------
LOG_FILE_NAME = 'log_sim_ppo_sb'


NN_MODEL_DATA_DIR  = sys.argv[1]  #'./model_data/rl_model/dagger or ppo' 

def _parse_optional_args(argv):
    opts = {
        "shield": "none",
        "shield_history": 5,
        "shield_quantile": 0.2,
        "shield_margin": 0.9,
        "shield_low_buffer_threshold": 10.0,
        "shield_low_buffer_margin": 0.8,
        "shield_buffer_guard": 1.0,
        "predictor_history": 15,
        "predictor_error_window": 5,
        "predictor_trend_window": 5,
        "predictor_volatility_z": 0.15,
        "predictor_trend_gain": 1.0,
        "external_predictor_file": "",
        "external_predictor_name": "",
        "external_predictor_fallback": "history_quantile",
        "external_predictor_max_age": 120.0,
    }
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--shield":
            opts["shield"] = argv[i + 1]
            i += 2
        elif arg == "--shield-history":
            opts["shield_history"] = int(argv[i + 1])
            i += 2
        elif arg == "--shield-quantile":
            opts["shield_quantile"] = float(argv[i + 1])
            i += 2
        elif arg == "--shield-margin":
            opts["shield_margin"] = float(argv[i + 1])
            i += 2
        elif arg == "--shield-low-buffer-threshold":
            opts["shield_low_buffer_threshold"] = float(argv[i + 1])
            i += 2
        elif arg == "--shield-low-buffer-margin":
            opts["shield_low_buffer_margin"] = float(argv[i + 1])
            i += 2
        elif arg == "--shield-buffer-guard":
            opts["shield_buffer_guard"] = float(argv[i + 1])
            i += 2
        elif arg == "--predictor-history":
            opts["predictor_history"] = int(argv[i + 1])
            i += 2
        elif arg == "--predictor-error-window":
            opts["predictor_error_window"] = int(argv[i + 1])
            i += 2
        elif arg == "--predictor-trend-window":
            opts["predictor_trend_window"] = int(argv[i + 1])
            i += 2
        elif arg == "--predictor-volatility-z":
            opts["predictor_volatility_z"] = float(argv[i + 1])
            i += 2
        elif arg == "--predictor-trend-gain":
            opts["predictor_trend_gain"] = float(argv[i + 1])
            i += 2
        elif arg == "--external-predictor-file":
            opts["external_predictor_file"] = argv[i + 1]
            i += 2
        elif arg == "--external-predictor-name":
            opts["external_predictor_name"] = argv[i + 1]
            i += 2
        elif arg == "--external-predictor-fallback":
            opts["external_predictor_fallback"] = argv[i + 1]
            i += 2
        elif arg == "--external-predictor-max-age":
            opts["external_predictor_max_age"] = float(argv[i + 1])
            i += 2
        else:
            raise ValueError(f"Unknown optional argument: {arg}")
    return opts


test_trace = TEST_TRACES
log_file_dir = LOG_FILE_DIR
# 判断是否提供了命令行参数
if len(sys.argv) >= 4:
    arg_test_trace = sys.argv[2]
    arg_log_file_dir = sys.argv[3]

    # 判断是否使用默认值
    if arg_test_trace == "default":
        test_trace = TEST_TRACES 
    else:
        test_trace = arg_test_trace
        assert isinstance(test_trace, str) and utils.is_valid_path(test_trace)
    
    if arg_log_file_dir == "default":
        log_file_dir = LOG_FILE_DIR 
    else:
        log_file_dir = arg_log_file_dir
        assert isinstance(test_trace, str) and utils.is_valid_path(test_trace)
else:
    print("Using default paths for test_trace and log_file_dir.")

SHIELD_OPTS = _parse_optional_args(sys.argv[4:])

print(test_trace, log_file_dir)
print("shield options:", SHIELD_OPTS)
EXTERNAL_PREDICTOR_TABLE = None
# ----------------------------
NN_MODEL = os.path.join(NN_MODEL_DATA_DIR, 'abr_model.zip')

os.makedirs(log_file_dir, exist_ok=True)
LOG_FILE = os.path.join(log_file_dir, LOG_FILE_NAME)

obs_stat = utils.load_obs_rms(NN_MODEL_DATA_DIR)
print('obs norm using state: ', obs_stat['norm_obs'])

# --------------------------------
def normalize_obs(obs, obs_stat):
    if obs_stat['norm_obs']:
        return np.clip(
            (obs - obs_stat['mean']) / np.sqrt(obs_stat['var'] + obs_stat['epsilon']),
            -obs_stat['clip_obs'],
            obs_stat['clip_obs']
        )
    else:
        return obs

def flatten_state(state: np.ndarray) -> np.ndarray:
    """
    把原始 (6,8) 状态铺平成长度 25 的向量：
      [ row0[-1], row1[-1], row2[:], row3[:], row4[:A_DIM], row5[-1] ]
    """
    s0 = state[0, -1:  ].reshape(1)
    s1 = state[1, -1:  ].reshape(1)
    s2 = state[2, :  ].reshape(S_LEN)
    s3 = state[3, :  ].reshape(S_LEN)
    s4 = state[4, :A_DIM].reshape(A_DIM)
    s5 = state[5, -1:  ].reshape(1)
    return np.concatenate([s0, s1, s2, s3, s4, s5], axis=0).astype(np.float32)
# --------------------------------------


def _safe_float(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_external_predictor_table(path, predictor_name=""):
    table = {}
    if not path:
        return table
    if not os.path.exists(path):
        raise FileNotFoundError(f"external predictor file not found: {path}")

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trace_id = row.get("trace_id") or row.get("trace_file") or row.get("trace")
            if not trace_id:
                continue
            row_predictor = row.get("predictor", "")
            if predictor_name and row_predictor and row_predictor != predictor_name:
                continue
            safe_cap = _safe_float(row.get("safe_cap_kbps"))
            if not np.isfinite(safe_cap) or safe_cap <= 0:
                continue
            time_s = _safe_float(row.get("time_s"))
            chunk_idx = int(_safe_float(row.get("chunk_idx"), -1))
            table.setdefault(os.path.basename(trace_id), []).append(
                {
                    "time_s": time_s,
                    "chunk_idx": chunk_idx,
                    "safe_cap_kbps": safe_cap,
                    "risk_score": _safe_float(row.get("risk_score")),
                    "predictor": row_predictor or predictor_name or "external_predictor",
                }
            )

    for rows in table.values():
        rows.sort(
            key=lambda item: (
                item["time_s"] if np.isfinite(item["time_s"]) else float("inf"),
                item["chunk_idx"],
            )
        )
    return table


def _get_external_predictor_table(opts):
    global EXTERNAL_PREDICTOR_TABLE
    if EXTERNAL_PREDICTOR_TABLE is None:
        EXTERNAL_PREDICTOR_TABLE = _load_external_predictor_table(
            opts.get("external_predictor_file", ""),
            opts.get("external_predictor_name", ""),
        )
        print(
            "external predictor traces:",
            len(EXTERNAL_PREDICTOR_TABLE),
            "file:",
            opts.get("external_predictor_file", ""),
            "name:",
            opts.get("external_predictor_name", ""),
        )
    return EXTERNAL_PREDICTOR_TABLE


def _lookup_external_safe_cap(trace_name, chunk_idx, time_s, opts):
    table = _get_external_predictor_table(opts)
    trace_key = os.path.basename(trace_name or "")
    rows = table.get(trace_key)
    if not rows:
        return float("nan"), "external_missing"

    if time_s is not None and np.isfinite(time_s):
        finite_rows = [row for row in rows if np.isfinite(row["time_s"])]
        if finite_rows:
            times = [row["time_s"] for row in finite_rows]
            pos = bisect_right(times, float(time_s)) - 1
            if pos >= 0:
                row = finite_rows[pos]
                max_age = float(opts.get("external_predictor_max_age", 120.0))
                if max_age < 0 or float(time_s) - row["time_s"] <= max_age:
                    return row["safe_cap_kbps"], row["predictor"]

    if chunk_idx is not None:
        chunk_rows = [row for row in rows if row["chunk_idx"] <= int(chunk_idx)]
        if chunk_rows:
            row = max(chunk_rows, key=lambda item: item["chunk_idx"])
            return row["safe_cap_kbps"], row["predictor"]

    return float("nan"), "external_missing"

def _harmonic_mean(values):
    values = [max(float(value), 1e-6) for value in values if np.isfinite(value)]
    if not values:
        return float("nan")
    return float(len(values) / np.sum([1.0 / value for value in values]))


def _update_prediction_error(predictor_state, actual_kbps, opts):
    last_prediction = predictor_state.get("last_prediction_kbps")
    if last_prediction is None or not np.isfinite(last_prediction) or actual_kbps <= 1e-6:
        return
    error = abs(float(last_prediction) - float(actual_kbps)) / float(actual_kbps)
    errors = predictor_state.setdefault("past_prediction_errors", [])
    errors.append(min(float(error), 10.0))
    max_len = max(int(opts["predictor_error_window"]) * 4, 16)
    if len(errors) > max_len:
        del errors[:-max_len]


def _predict_safe_cap_kbps(throughput_history_kbps, buffer_size, opts, predictor_state):
    history_len = max(int(opts["shield_history"]), 1)
    if opts["shield"] == "starnet_proxy":
        history_len = max(int(opts["predictor_history"]), history_len)
    history = np.asarray(throughput_history_kbps[-history_len:], dtype=np.float64)
    history = history[np.isfinite(history) & (history > 0)]
    if history.size == 0:
        return float("nan"), "none"

    if opts["shield"] in ("lower_quantile", "download_time"):
        safe_cap_kbps = float(np.quantile(history, opts["shield_quantile"]))
        predictor_state["last_prediction_kbps"] = safe_cap_kbps
        predictor_name = "history_quantile"
    elif opts["shield"] == "rmpc_predictive":
        _update_prediction_error(predictor_state, history[-1], opts)
        rmpc_history = history[-max(int(opts["shield_history"]), 1):]
        harmonic_bandwidth = _harmonic_mean(rmpc_history)
        errors = predictor_state.get("past_prediction_errors", [])
        error_window = max(int(opts["predictor_error_window"]), 1)
        max_error = max(errors[-error_window:]) if errors else 0.0
        safe_cap_kbps = harmonic_bandwidth / (1.0 + max_error)
        predictor_state["last_prediction_kbps"] = harmonic_bandwidth
        predictor_name = "rmpc_harmonic_error"
    elif opts["shield"] == "starnet_proxy":
        # A lightweight StarNet-style lower-bound proxy:
        # recent quantile + downward trend + volatility budget. This keeps the
        # online interface ready for a trained StarNet risk predictor.
        trend_window = max(int(opts["predictor_trend_window"]), 1)
        short = history[-trend_window:]
        prev = history[-2 * trend_window : -trend_window]
        quantile_bound = float(np.quantile(history, opts["shield_quantile"]))
        center = float(np.median(short))
        trend = 0.0
        if prev.size > 0:
            trend = center - float(np.median(prev))
        volatility = float(np.std(history))
        trend_penalty = max(-trend, 0.0) * float(opts["predictor_trend_gain"])
        risk_bound = center - trend_penalty - float(opts["predictor_volatility_z"]) * volatility
        safe_cap_kbps = min(quantile_bound, risk_bound)
        predictor_state["last_prediction_kbps"] = max(risk_bound, 0.0)
        predictor_name = "starnet_proxy_lower_bound"
    else:
        raise ValueError(f"Unsupported shield mode: {opts['shield']}")

    safe_cap_kbps *= float(opts["shield_margin"])
    if buffer_size < opts["shield_low_buffer_threshold"]:
        safe_cap_kbps *= float(opts["shield_low_buffer_margin"])
    safe_cap_kbps = max(safe_cap_kbps, float(VIDEO_BIT_RATE[0]))
    return safe_cap_kbps, predictor_name


def _predict_external_or_fallback_safe_cap(
    trace_name,
    chunk_idx,
    time_s,
    throughput_history_kbps,
    buffer_size,
    opts,
    predictor_state,
):
    safe_cap_kbps, predictor_name = _lookup_external_safe_cap(
        trace_name,
        chunk_idx,
        time_s,
        opts,
    )
    if not np.isfinite(safe_cap_kbps):
        fallback = opts.get("external_predictor_fallback", "history_quantile")
        if fallback in ("none", "off", ""):
            return safe_cap_kbps, predictor_name
        if throughput_history_kbps:
            fallback_opts = dict(opts)
            fallback_opts["shield"] = "download_time"
            safe_cap_kbps, fallback_name = _predict_safe_cap_kbps(
                throughput_history_kbps,
                buffer_size,
                fallback_opts,
                predictor_state,
            )
            predictor_name = f"fallback_{fallback_name}"
    else:
        safe_cap_kbps *= float(opts["shield_margin"])
        if buffer_size < opts["shield_low_buffer_threshold"]:
            safe_cap_kbps *= float(opts["shield_low_buffer_margin"])
        safe_cap_kbps = max(safe_cap_kbps, float(VIDEO_BIT_RATE[0]))
        predictor_state["last_prediction_kbps"] = safe_cap_kbps
    return safe_cap_kbps, predictor_name


def choose_shielded_action(
    raw_action,
    throughput_history_kbps,
    next_video_chunk_sizes,
    buffer_size,
    opts,
    predictor_state,
    trace_name=None,
    chunk_idx=None,
    time_s=None,
):
    if opts["shield"] == "none":
        return raw_action, {
            "safe_cap_kbps": float("nan"),
            "shielded": False,
            "predictor_name": "none",
        }
    supported_shields = (
        "lower_quantile",
        "download_time",
        "rmpc_predictive",
        "starnet_proxy",
        "external_predictor",
    )
    if opts["shield"] not in supported_shields:
        raise ValueError(f"Unsupported shield mode: {opts['shield']}")
    if not throughput_history_kbps and opts["shield"] != "external_predictor":
        return raw_action, {
            "safe_cap_kbps": float("nan"),
            "shielded": False,
            "predictor_name": "none",
        }

    if opts["shield"] == "external_predictor":
        safe_cap_kbps, predictor_name = _predict_external_or_fallback_safe_cap(
            trace_name,
            chunk_idx,
            time_s,
            throughput_history_kbps,
            buffer_size,
            opts,
            predictor_state,
        )
    else:
        safe_cap_kbps, predictor_name = _predict_safe_cap_kbps(
            throughput_history_kbps,
            buffer_size,
            opts,
            predictor_state,
        )
    if not np.isfinite(safe_cap_kbps):
        return raw_action, {
            "safe_cap_kbps": float("nan"),
            "shielded": False,
            "predictor_name": predictor_name,
        }

    safe_action = 0
    max_candidate = min(raw_action, A_DIM - 1)
    usable_buffer = max(buffer_size - opts["shield_buffer_guard"], 0.25)
    for action in range(max_candidate + 1):
        estimated_download_s = next_video_chunk_sizes[action] * 8.0 / safe_cap_kbps / M_IN_K
        bitrate_ok = VIDEO_BIT_RATE[action] <= safe_cap_kbps
        buffer_ok = estimated_download_s <= usable_buffer
        if opts["shield"] == "lower_quantile":
            safe = bitrate_ok and buffer_ok
        else:
            safe = buffer_ok
        if safe:
            safe_action = action
    return int(safe_action), {
        "safe_cap_kbps": safe_cap_kbps,
        "shielded": int(safe_action) < int(raw_action),
        "predictor_name": predictor_name,
    }

def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    # 加载测试轨迹
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(test_trace)

    # 初始化环境
    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    # 日志文件
    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')

    # 加载 PPO 模型
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = PPO.load(NN_MODEL, device=device)
    # 将 policy 切换到 eval 模式
    model.policy.eval()
    model.policy.to(device)
    print("Testing model restored from", NN_MODEL)

    # # ─── 新增：加载训练时保存的 VecNormalize wrapper ───
    # # 注意路径要和训练时 save() 所用的一致

    # # 创建一个 1-env 的 DummyVecEnv，其 obs_shape 应和训练时一致
    # dummy_env = DummyVecEnv([
    #     lambda: ABRGymEnv(trace_path=TRAIN_TRACES, expert_algo='bs')
    # ])
    # # 传入 dummy_env，让 load 成功设置 num_envs
    # norm_path = "./model_data/vec_normalize.pkl"
    # norm_wrapper = VecNormalize.load(norm_path, dummy_env)
    # # 评估／测试时关闭对 reward 的归一化、也不再更新统计
    # norm_wrapper.training = False
    # norm_wrapper.norm_reward = False
    # ──────────────────────────────────────────────────────

    # 初始化循环变量
    time_stamp = 0
    trace_time_stamp = 0
    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    s_batch = [np.zeros((S_INFO, S_LEN))]
    r_batch = []
    entropy_record = []
    entropy_ = 0.5
    video_count = 0
    throughput_history_kbps = []
    predictor_state = {}
    raw_bit_rate = DEFAULT_QUALITY
    shield_safe_cap_kbps = float("nan")
    shielded_action = False

    while True:
        # 获取视频块
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(bit_rate)
        if delay > 0:
            throughput_history_kbps.append(video_chunk_size * 8.0 / delay)

        time_stamp += delay
        time_stamp += sleep_time
        trace_time_stamp += delay
        trace_time_stamp += sleep_time

        # 计算 reward
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                 - REBUF_PENALTY * rebuf \
                 - SMOOTH_PENALTY * abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
        r_batch.append(reward)
        last_bit_rate = bit_rate

        # 写日志
        log_file.write(
            f"{time_stamp / M_IN_K}\t"
            f"{VIDEO_BIT_RATE[bit_rate]}\t"
            f"{buffer_size}\t"
            f"{rebuf}\t"
            f"{video_chunk_size}\t"
            f"{delay}\t"
            f"{entropy_}\t"
            f"{reward}\t"
            f"{raw_bit_rate}\t"
            f"{shield_safe_cap_kbps}\t"
            f"{int(shielded_action)}\n"
        )
        log_file.flush()

        # 构造下一个 state
        prev_state = s_batch[-1] if s_batch else np.zeros((S_INFO, S_LEN))
        state = np.roll(prev_state, -1, axis=1)
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR
        state[5, -1] = min(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / CHUNK_TIL_VIDEO_END_CAP
        if NORMALIZED:
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K / 10.
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K / 10.
        else:
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K

        # 用 PPO 模型预测
        flat_state = flatten_state(state)
        # 假设 obs_raw 是你直接从环境 / 数据源拿到的原始观测
        normed_flat_state = normalize_obs(flat_state, obs_stat)
        # normed_flat_state = norm_wrapper.normalize_obs(flat_state[np.newaxis, ...])
        # 去掉 batch 维度，继续后面的 flatten
        # normed_flat_state = normed_flat_state[0]
        obs_tensor = torch.from_numpy(normed_flat_state).to(device).unsqueeze(0)
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.cpu().detach().numpy()[0]
        entropy_ = dist.distribution.entropy().cpu().detach().numpy()[0]

        raw_bit_rate = int(np.argmax(probs))
        current_trace_name = all_file_names[net_env.trace_idx]
        next_chunk_idx = int(net_env.video_chunk_counter)
        bit_rate, shield_diag = choose_shielded_action(
            raw_bit_rate,
            throughput_history_kbps,
            next_video_chunk_sizes,
            buffer_size,
            SHIELD_OPTS,
            predictor_state,
            trace_name=current_trace_name,
            chunk_idx=next_chunk_idx,
            time_s=trace_time_stamp / M_IN_K,
        )
        shield_safe_cap_kbps = shield_diag["safe_cap_kbps"]
        shielded_action = shield_diag["shielded"]

        # 存储
        s_batch.append(state)
        entropy_record.append(entropy_)

        # 视频结束处理
        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY
            s_batch = [np.zeros((S_INFO, S_LEN))]
            r_batch = []
            entropy_record = []
            throughput_history_kbps = []
            predictor_state = {}
            trace_time_stamp = 0
            raw_bit_rate = DEFAULT_QUALITY
            shield_safe_cap_kbps = float("nan")
            shielded_action = False

            video_count += 1
            if video_count >= len(all_file_names):
                break

            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')

    log_file.close()
    print("Testing completed.")

if __name__ == '__main__':
    main()
