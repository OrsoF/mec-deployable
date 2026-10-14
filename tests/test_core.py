import json
from pathlib import Path

import numpy as np
import pytest

from network_ppo.constraints import assess
from network_ppo.heuristic import feasible_seed, greedy_uplift
from network_ppo.instance_io import load_instance
from network_ppo.paths import PathCatalog
from network_ppo.problem import Solution, validate_compatible
from network_ppo.solution_io import decision_json

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "instances" / "train"
HELD_OUT = ROOT / "instances" / "eval" / "bottleneck_unseen.json"


def test_exactly_ten_training_instances_plus_unseen_compatible():
    scenarios = [load_instance(p) for p in sorted(TRAIN.glob("*.json"))]
    assert len(scenarios) == 10
    signature = validate_compatible(scenarios)
    assert validate_compatible([load_instance(HELD_OUT)], signature) == signature
    assert len(signature["arcs"]) == 14


@pytest.mark.parametrize("file", sorted(TRAIN.glob("*.json")) + [HELD_OUT])
def test_seed_and_greedy_reference_are_feasible(file):
    p = load_instance(file)
    routes = PathCatalog(p, 3)
    seed = feasible_seed(p, routes)
    assert assess(p, seed, routes).feasible
    assert np.array_equal(seed.quantities, [dm.min_mbps for dm in p.demands])
    ref = greedy_uplift(p, routes, seed)
    report = assess(p, ref, routes)
    assert report.feasible
    assert report.total_mbps >= seed.quantities.sum()
    output = decision_json(p, ref, routes, checkpoint="test", policy_steps=0, seed=1)
    assert output["feasible"]
    assert len(output["placements"]) == len(p.services)
    assert len(output["demands"]) == len(p.demands)
    assert len(output["directed_links"]) == 2 * len(p.links)
    assert all(d["latency_ms"] <= d["latency_budget_ms"] for d in output["demands"])
    assert abs(sum(d["quantity_mbps"] for d in output["demands"]) - output["total_served_mbps"]) < 1e-8


def _mutated(tmp_path, mutate):
    data = json.loads((TRAIN / "bottleneck_00.json").read_text())
    mutate(data)
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize("key,value", [
    ("capacity_mbps", -1), ("capacity_mbps", float("nan")), ("capacity_mbps", True),
])
def test_reject_bad_link_capacity(tmp_path, key, value):
    path = _mutated(tmp_path, lambda data: data["links"][0].update({key: value}))
    with pytest.raises(ValueError, match="capacity_mbps"):
        load_instance(path)


def test_reject_invalid_flow_bounds(tmp_path):
    path = _mutated(tmp_path, lambda data: data["services"][0]["demands"][0].update(
        {"min_mbps": 9, "max_mbps": 2}))
    with pytest.raises(ValueError, match="min_mbps exceeds max_mbps"):
        load_instance(path)


def test_reject_fractional_flows(tmp_path):
    path = _mutated(tmp_path, lambda data: data["services"][0]["demands"][0].update(
        {"max_mbps": 2.5}))
    with pytest.raises(ValueError, match="integer"):
        load_instance(path)


def test_reject_topology_reordering(tmp_path):
    path = _mutated(tmp_path, lambda data: data["nodes"].reverse())
    expected = load_instance(TRAIN / "bottleneck_00.json").signature
    with pytest.raises(ValueError, match="Incompatible"):
        validate_compatible([load_instance(path)], expected)


def test_reject_unknown_origin(tmp_path):
    path = _mutated(tmp_path, lambda data: data["services"][0]["demands"][0].update(
        {"origin": "nonexistent"}))
    with pytest.raises(ValueError, match="unknown node"):
        load_instance(path)


def test_reject_duplicate_undirected_link(tmp_path):
    def duplicate(data):
        duplicated = data["links"][0].copy()
        duplicated["source"], duplicated["target"] = duplicated["target"], duplicated["source"]
        data["links"].append(duplicated)
    path = _mutated(tmp_path, duplicate)
    with pytest.raises(ValueError, match="duplicate"):
        load_instance(path)


def test_reject_latency_constraint_violation():
    from dataclasses import replace
    p = load_instance(HELD_OUT)
    p = replace(p, demands=tuple(replace(d, max_latency_ms=6.0) for d in p.demands))
    catalog = PathCatalog(p, 3)
    seed = feasible_seed(p, catalog)
    assert assess(p, seed, catalog).feasible
    moved = seed.copy()
    moved.placement[:] = len(p.nodes) - 1   # cloud violates stringent latency
    report = assess(p, moved, catalog)
    assert not report.feasible
    assert any("latency" in message for message in report.violations)


def test_reject_cpu_capacity_violation():
    p = load_instance(HELD_OUT)
    catalog = PathCatalog(p, 3)
    seed = feasible_seed(p, catalog)
    moved = seed.copy()
    moved.placement[:] = 0     # gateway has zero CPU
    report = assess(p, moved, catalog)
    assert not report.feasible
    assert any("CPU" in message for message in report.violations)


def test_reject_quantity_out_of_bounds():
    p = load_instance(HELD_OUT)
    catalog = PathCatalog(p, 3)
    seed = feasible_seed(p, catalog)
    seed.quantities[0] = 500
    report = assess(p, seed, catalog)
    assert not report.feasible
    assert any("quantity" in message for message in report.violations)


def test_directed_arc_loads_are_independent():
    p = load_instance(HELD_OUT)
    cat = PathCatalog(p, 3)
    seed = feasible_seed(p, cat)
    loads = assess(p, seed, cat).edge_load
    # Traffic from gateways to edge, not symmetric mirroring into reverse arcs.
    assert any(loads[::2] != loads[1::2])


def test_export_refuses_infeasible():
    p = load_instance(HELD_OUT)
    cat = PathCatalog(p, 3)
    seed = feasible_seed(p, cat)
    seed.quantities[0] = 1000
    with pytest.raises(ValueError, match="infeasible"):
        decision_json(p, seed, cat, checkpoint="x", policy_steps=1, seed=0)


def test_same_node_route_is_zero_hop():
    p = load_instance(HELD_OUT)
    c = PathCatalog(p, 3)
    for k in range(3):
        assert c.route(0, 0, k) == (0,)
