"""Deterministic, split-separated temporal network workloads.

The public forecast is constructed from the deterministic trend BEFORE realized
traffic noise is drawn. Distinct seeds and workload regimes test robustness;
there is no use of held-out data in training or hyperparameter selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "instances/eval/bottleneck_unseen.json"


def make_case(base: dict, *, split: str, seed: int, periods: int = 24,
              regime: str = "daily") -> dict:
    rng = np.random.default_rng(seed)
    original = np.array([d["max_mbps"] for s in base["services"] for d in s["demands"]], dtype=float)
    lower = np.array([d["min_mbps"] for s in base["services"] for d in s["demands"]], dtype=int)
    d = len(original)
    t = np.arange(periods)
    phase = rng.uniform(0, 2 * np.pi, size=d)
    baseline = rng.uniform(0.55, 0.85, d) * original
    amplitude = rng.uniform(.12, .35, d) * original
    trend = baseline[None, :] + amplitude[None, :] * np.sin(
        2 * np.pi * t[:, None] / 16 + phase[None, :])
    if regime in ("bursty", "stress"):
        for _ in range(2 if regime == "bursty" else 5):
            start = int(rng.integers(0, periods))
            length = int(rng.integers(1, 4))
            subset = int(rng.integers(d))
            trend[start:min(periods, start + length), subset] += 2 if regime == "bursty" else 4
    # Demand innovations are independent of published forecast, with persistence.
    innovations = np.zeros_like(trend)
    noise = .45 if regime != "stress" else .9
    for i in range(1, periods):
        innovations[i] = .55 * innovations[i - 1] + rng.normal(0, noise, size=d)
    forecast = np.maximum(lower, np.rint(trend)).astype(int)
    realized = np.maximum(lower, np.rint(trend + innovations)).astype(int)
    # Mild contemporaneous link status shifts. Future values never enter MPC.
    factors = np.ones((periods, len(base["links"])), dtype=float)
    if regime == "stress":
        # A failure on one cross link (already only 2 Mbps) is safe for min flows.
        factors[periods // 2:periods // 2 + 3, 1] = .75
    return {
        "schema_version": 1,
        "id": f"temporal_{split}_{regime}_{seed}",
        "split": split,
        "description": "Synthetic temporal traffic; forecast does not reveal realized future noise",
        "base_instance": base,
        "realized_mbps": realized.tolist(),
        "forecast_mbps": forecast.tolist(),
        "capacity_scale": factors.tolist(),
        "migration_cost": 1.0,
        "route_change_cost": 0.3,
        "active_node_cost": 0.1,
        "discount": 0.98,
        "regime": regime,
        "generation_seed": seed,
    }


def generate(root: Path, *, periods: int = 24) -> None:
    base = json.loads(BASE.read_text(encoding="utf-8"))
    configs = {
        "train": [(100 + i, "daily" if i % 3 else "bursty") for i in range(12)],
        "validation": [(400 + i, "daily" if i % 2 else "bursty") for i in range(4)],
        "test": [(900 + i, "daily" if i == 0 else "bursty" if i == 1 else "stress") for i in range(4)],
    }
    for split, jobs in configs.items():
        folder = root / split
        folder.mkdir(parents=True, exist_ok=True)
        for seed, regime in jobs:
            out = make_case(base, split=split, seed=seed, regime=regime, periods=periods)
            (folder / f"{out['id']}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "instances/temporal")
    parser.add_argument("--periods", type=int, default=24)
    args = parser.parse_args()
    if args.periods < 2:
        parser.error("periods must be >= 2")
    generate(args.output, periods=args.periods)
