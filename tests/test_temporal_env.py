"""Optional Gymnasium integration; executed in RL CI, skipped without Gymnasium."""

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
from pathlib import Path

from gymnasium.utils.env_checker import check_env

from network_ppo.dynamics import load_temporal_scenario
from network_ppo.temporal_env import TemporalNetworkEnv

ROOT = Path(__file__).resolve().parents[1]


def test_temporal_env_is_gymnasium_compliant():
    scenario = load_temporal_scenario(
        ROOT / "instances/temporal/test/temporal_test_daily_900.json"
    )
    env = TemporalNetworkEnv(scenario)
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=7)
    assert obs.shape == env.observation_space.shape
    assert info["scenario"] == scenario.id
    action = np.concatenate((env.previous.placement, env.previous.routes)).astype(int)
    _, reward, terminated, truncated, info = env.step(action)
    assert info["valid_proposal"]
    assert not terminated and not truncated
    assert info["migrations"] == 0 and info["route_changes"] == 0
    assert reward == pytest.approx(info["reward_without_invalid_penalty"])
    for _ in range(scenario.horizon - 1):
        action = np.concatenate((env.previous.placement, env.previous.routes)).astype(
            int
        )
        _, _, terminated, _, _ = env.step(action)
    assert terminated
    env.close()
