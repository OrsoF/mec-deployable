"""Tiny integer allocation problems with provable optimality."""

from __future__ import annotations

import pytest

pytest.importorskip("scipy")

from network_ppo.exact import solve_exact_small
from network_ppo.paths import PathCatalog
from network_ppo.problem import Demand, Link, Node, Problem, Service


def test_exact_nontrivial_bounded_capacity():
    case = Problem(
        "tiny",
        (Node("gateway", 0), Node("edge", 8)),
        (Link(0, 1, 4, 2, 0.1),),
        (Service("video", 4),),
        (Demand("camera", 0, 0, 1, 6, 10),),
    )
    report = solve_exact_small(case, PathCatalog(case, 1), max_configs=10)
    assert report.certified
    assert int(report.solution.quantities.sum()) == 4
    assert report.configurations_checked > 0


def test_exact_certificates_from_upper_bound():
    from pathlib import Path

    from network_ppo.instance_io import load_instance

    case = load_instance(
        Path(__file__).resolve().parents[1] / "instances/eval/bottleneck_unseen.json"
    )
    result = solve_exact_small(case, PathCatalog(case, 3), max_configs=1)
    assert result.certified and result.configurations_checked == 0
    assert result.solution.quantities.sum() == 28
