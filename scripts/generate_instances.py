"""Reproduce the deterministic congested train/eval split.

The test parameters are intentionally outside the training grid's precise
combinations; topology and identifiers stay fixed for policy compatibility.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "instances/templates/bottleneck_base.json"


def build(base: dict, *, name: str, cross: int, local: int, core: int,
          demand_max: int, latency: float) -> dict:
    raw = copy.deepcopy(base)
    raw["id"] = name
    raw["description"] = ("Synthetic routing bottleneck; independent directional "
                          "capacities, feasible core detours, fixed topology")
    for i, value in {0: local, 1: cross, 2: cross, 3: local, 4: core, 5: core}.items():
        raw["links"][i]["capacity_mbps"] = value
    for service in raw["services"]:
        for demand in service["demands"]:
            demand["max_mbps"] = demand_max
            demand["max_latency_ms"] = latency
    return raw


def generate(output: Path) -> None:
    base = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    train = output / "train"
    evaluation = output / "eval"
    train.mkdir(parents=True, exist_ok=True)
    evaluation.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        params = {
            "cross": 1 + (i % 4), "local": 12 + (i % 5),
            "core": 18 + (i % 4), "demand_max": 5 + (i % 4),
            "latency": 15 + (i % 6),
        }
        raw = build(base, name=f"bottleneck_train_{i:02d}", **params)
        (train / f"bottleneck_{i:02d}.json").write_bytes(
            (json.dumps(raw, indent=2) + "\n").encode("utf-8"))
    held = build(base, name="routing_bottleneck_heldout",
                 cross=2, local=14, core=20, demand_max=7, latency=25)
    (evaluation / "bottleneck_unseen.json").write_bytes(
        (json.dumps(held, indent=2) + "\n").encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "instances")
    generate(parser.parse_args().output)
