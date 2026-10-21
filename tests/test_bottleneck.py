"""Regression, reproducibility, integration, and fail-closed behavior tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from network_ppo.constraints import assess
from network_ppo.heuristic import feasible_seed, solve_baseline
from network_ppo.inference import DecisionEngine
from network_ppo.instance_io import load_instance, parse_instance
from network_ppo.paths import PathCatalog
from network_ppo.problem import validate_compatible

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "instances/eval/bottleneck_unseen.json"


def test_congested_baselines_reproduce_optimality_certificate():
    case = load_instance(EVAL)
    routes = PathCatalog(case, 3)
    seed = feasible_seed(case, routes)
    greedy = solve_baseline(case, routes, "greedy", seed)
    lookahead = solve_baseline(case, routes, "lookahead", seed)
    multi = solve_baseline(case, routes, "multistart", seed)
    assert int(greedy.quantities.sum()) == 18
    assert list(greedy.quantities) == [7, 2, 2, 7]
    assert int(lookahead.quantities.sum()) == 28
    assert int(multi.quantities.sum()) == 28
    assert assess(case, lookahead, routes).feasible
    assert int(lookahead.quantities.sum()) == sum(d.max_mbps for d in case.demands)


def test_training_instances_have_bottlenecks_and_holdout_distinct():
    train_files = sorted((ROOT / "instances/train").glob("*.json"))
    assert len(train_files) == 10
    cases = [load_instance(x) for x in train_files]
    sig = validate_compatible(cases)
    assert validate_compatible([load_instance(EVAL)], sig) == sig
    # These are distinct parameterizations, not duplicated test examples.
    assert len({x.read_bytes() for x in train_files + [EVAL]}) == 11
    for case in cases:
        paths = PathCatalog(case, 3)
        assert (
            solve_baseline(case, paths, "lookahead").quantities.sum()
            > solve_baseline(case, paths, "greedy").quantities.sum()
        )


def test_generator_reproduces_committed_instances(tmp_path):
    from scripts.generate_instances import generate

    generate(tmp_path)
    for file in sorted((ROOT / "instances/train").glob("*.json")) + [EVAL]:
        rel = file.relative_to(ROOT / "instances")
        assert (tmp_path / rel).read_bytes() == file.read_bytes()


def test_inference_json_feasibility_and_algorithm_provenance():
    case = load_instance(EVAL)
    engine = DecisionEngine(strategy="auto", seed=5)
    out = engine.decide(case)
    assert out["feasible"] is True
    assert out["selected_method"] == "lookahead"
    assert out["candidate_scores_mbps"] == {"lookahead": 28.0}
    assert out["certified_optimal"] is True
    assert out["total_served_mbps"] == 28
    assert out["decision_wall_ms"] >= 0
    json.dumps(out, allow_nan=False)


def test_auto_records_broken_checkpoint_and_falls_back(tmp_path):
    case = load_instance(EVAL)
    engine = DecisionEngine(strategy="auto", checkpoint=tmp_path / "missing")
    result = engine.decide(case)
    assert result["feasible"]
    assert result["selected_method"] == "lookahead"
    assert result["warnings"]


def test_ppo_mode_requires_checkpoint():
    with pytest.raises(ValueError, match="checkpoint"):
        DecisionEngine(strategy="ppo")


def test_parse_instance_rejects_bad_units_and_nonobjects():
    raw = json.loads(EVAL.read_text())
    with pytest.raises(ValueError, match="schema_version"):
        parse_instance([])
    raw["units"]["flow"] = "Kbps"
    with pytest.raises(ValueError, match="units"):
        parse_instance(raw)


def test_deploy_cli_from_clean_process(tmp_path):
    output = tmp_path / "decision.json"
    run = subprocess.run(
        [
            sys.executable,
            "deploy.py",
            "--instance",
            str(EVAL),
            "--strategy",
            "lookahead",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert run.returncode == 0, run.stderr
    data = json.loads(output.read_text())
    assert data["feasible"] and data["total_served_mbps"] == 28


def test_http_json_api_integration():
    from http.server import ThreadingHTTPServer
    from threading import Thread
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    from serve import handler_factory

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_factory(DecisionEngine()))
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(url + "/ready", timeout=3) as resp:
            assert json.load(resp)["status"] == "ready"
        raw = EVAL.read_bytes()
        with urlopen(
            Request(
                url + "/decide", data=raw, headers={"Content-Type": "application/json"}
            ),
            timeout=5,
        ) as resp:
            out = json.load(resp)
            assert out["total_served_mbps"] == 28
            assert out["feasible"]
        try:
            urlopen(
                Request(
                    url + "/decide",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                ),
                timeout=3,
            )
        except HTTPError as err:
            assert err.code == 400
        else:
            pytest.fail("Malformed instance unexpectedly accepted")
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)
