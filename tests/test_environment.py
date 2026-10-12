"""Gymnasium contract tests. Skipped if optional runtime not installed locally."""
import numpy as np
import pytest

gymnasium = pytest.importorskip("gymnasium")

from network_ppo.environment import NetworkPPOEnv
from network_ppo.instance_io import load_instance
from network_ppo.constraints import assess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env():
    case = load_instance(ROOT / "instances/eval/factory_shift_unseen.json")
    return NetworkPPOEnv(case, episode_steps=8, paths=3)


def test_actions_preserve_feasibility_and_good_shape():
    env = _env()
    try:
        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape
        assert env.observation_space.contains(obs)
        rng = np.random.default_rng(42)
        for _ in range(8):
            o, reward, terminated, truncated, step_info = env.step(int(rng.integers(env.action_space.n)))
            assert o.shape == obs.shape
            assert np.isfinite(reward)
            assert env.observation_space.contains(o)
            assert assess(env.problem, env.state, env.catalogue).feasible
            assert not terminated
        assert truncated
    finally:
        env.close()


def test_infeasible_action_rejected_and_reward_penalized():
    env = _env()
    env.reset(seed=2)
    try:
        old = env.state.copy()
        # Set first service to node 0 (CPU=0), which is forbidden.
        _, reward, _, _, info = env.step(0)
        assert not info["feasible_action"]
        assert reward < 0
        assert np.array_equal(env.state.placement, old.placement)
    finally:
        env.close()


def test_positive_throughput_action_reward():
    env = _env()
    env.reset(seed=0)
    try:
        # Find one feasible +1 edit in the current min-Mbps starting point.
        q_base = env.placement_actions + env.routing_actions
        candidates = [q_base + 2 * d + 1 for d in range(len(env.problem.demands))]
        assert any(assess(env.problem, _increment(env.state, d), env.catalogue).feasible
                   for d in range(len(candidates)))
        for action in candidates:
            _, reward, _, _, info = env.step(action)
            if info["feasible_action"]:
                assert reward == 1.0
                assert env.best_episode.quantities.sum() > env.seeds[0].quantities.sum()
                break
    finally:
        env.close()


def _increment(solution, demand):
    copy = solution.copy()
    copy.quantities[demand] += 1
    return copy


def test_training_instance_sampling_is_reproducible():
    root = ROOT / "instances/train"
    cases = [load_instance(path) for path in sorted(root.glob("*.json"))]
    env = NetworkPPOEnv(cases)
    try:
        series = [env.reset(seed=123)[1]["instance_id"]]
        series += [env.reset()[1]["instance_id"] for _ in range(15)]
        again = [env.reset(seed=123)[1]["instance_id"]]
        again += [env.reset()[1]["instance_id"] for _ in range(15)]
        assert series == again
        assert len(set(series)) > 1
    finally:
        env.close()
