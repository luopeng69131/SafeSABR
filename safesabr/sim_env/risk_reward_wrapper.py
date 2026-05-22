from collections import deque

import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper


class RiskAwareRewardWrapper(VecEnvWrapper):
    """
    Adds a training-time tail-risk penalty to ABR rewards.

    The wrapper leaves observations/dones untouched and only modifies rewards
    seen by PPO. Evaluation scripts still report the original QoE reward.
    """

    def __init__(
        self,
        venv,
        mode="none",
        alpha=0.9,
        penalty_lambda=1.0,
        budget=0.0,
        window=512,
    ):
        super().__init__(venv)
        self.mode = mode
        self.alpha = float(alpha)
        self.penalty_lambda = float(penalty_lambda)
        self.budget = float(budget)
        self.window = int(window)
        self.rebuffer_history = deque(maxlen=max(self.window, 1))
        self.last_penalty_mean = 0.0
        self.last_cvar = 0.0
        self.last_var = 0.0

        self._validate_config()

    def set_risk_config(
        self,
        mode=None,
        alpha=None,
        penalty_lambda=None,
        budget=None,
        window=None,
        reset_history=False,
    ):
        if mode is not None:
            self.mode = mode
        if alpha is not None:
            self.alpha = float(alpha)
        if penalty_lambda is not None:
            self.penalty_lambda = float(penalty_lambda)
        if budget is not None:
            self.budget = float(budget)
        if window is not None:
            self.window = int(window)
        self._validate_config()
        if reset_history:
            self.rebuffer_history = deque(maxlen=max(self.window, 1))
            self.last_penalty_mean = 0.0
            self.last_cvar = 0.0
            self.last_var = 0.0

    def _validate_config(self):
        if self.mode not in ("none", "mean_rebuf", "cvar_rebuf"):
            raise ValueError(f"Unsupported risk reward mode: {self.mode}")
        if not 0.0 <= self.alpha < 1.0:
            raise ValueError(f"risk alpha must be in [0, 1), got {self.alpha}")
        if self.window < 1:
            raise ValueError(f"risk window must be >= 1, got {self.window}")

    def reset(self):
        return self.venv.reset()

    def step_async(self, actions):
        return self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        if self.mode == "none" or self.penalty_lambda <= 0:
            return obs, rewards, dones, infos

        rewards = np.array(rewards, dtype=np.float32, copy=True)
        rebufs = np.array(
            [float(info.get("rebuffer", 0.0) or 0.0) for info in infos],
            dtype=np.float32,
        )
        for value in rebufs:
            self.rebuffer_history.append(float(value))

        penalties = self._compute_penalty(rebufs)
        rewards -= penalties.astype(np.float32)

        self.last_penalty_mean = float(np.mean(penalties)) if len(penalties) else 0.0
        for i, info in enumerate(infos):
            info["risk_penalty"] = float(penalties[i])
            info["risk_adjusted_reward"] = float(rewards[i])
            info["risk_mode"] = self.mode
            info["risk_var_rebuf"] = float(self.last_var)
            info["risk_cvar_rebuf"] = float(self.last_cvar)

        return obs, rewards, dones, infos

    def get_risk_state(self):
        history = np.array(self.rebuffer_history, dtype=np.float32)
        return {
            "mode": self.mode,
            "alpha": self.alpha,
            "lambda": self.penalty_lambda,
            "budget": self.budget,
            "window": self.window,
            "history_size": int(history.size),
            "history_mean_rebuffer": float(np.mean(history)) if history.size else 0.0,
            "history_p95_rebuffer": float(np.percentile(history, 95)) if history.size else 0.0,
            "last_penalty_mean": self.last_penalty_mean,
            "last_var_rebuffer": self.last_var,
            "last_cvar_rebuffer": self.last_cvar,
        }

    def _compute_penalty(self, rebufs):
        if self.mode == "mean_rebuf":
            excess = np.maximum(rebufs - self.budget, 0.0)
            self.last_var = float(self.budget)
            self.last_cvar = float(np.mean(excess + self.budget)) if len(excess) else 0.0
            return self.penalty_lambda * excess

        history = np.array(self.rebuffer_history, dtype=np.float32)
        if history.size == 0:
            return np.zeros_like(rebufs, dtype=np.float32)

        var = float(np.quantile(history, self.alpha))
        tail_count = max(1, int(np.ceil((1.0 - self.alpha) * history.size)))
        cvar = float(np.mean(np.sort(history)[-tail_count:]))
        threshold = max(self.budget, var)
        excess = np.maximum(rebufs - threshold, 0.0)

        self.last_var = var
        self.last_cvar = cvar
        return self.penalty_lambda * excess
