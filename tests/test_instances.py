from pathlib import Path

import pytest

from mec_ppo.instances import load_instance, validate_compatible
from mec_ppo.results import format_result
from mecbench.solvers.fast_heuristic import Solver as FastHeuristic


ROOT = Path(__file__).resolve().parents[1]


def test_training_pack_is_compatible_and_feasible():
    examples = [load_instance(p) for p in sorted((ROOT / 'instances/train').glob('*.json'))]
    assert len(examples) == 10
    signature = validate_compatible(examples)
    target = load_instance(ROOT / 'instances/eval/factory_shift_unseen.json')
    assert validate_compatible([target], signature) == signature
    assert all(x.problem.data.n_demands == 4 for x in examples)


def test_reference_heuristic_and_decision_export():
    example = load_instance(ROOT / 'instances/eval/factory_shift_unseen.json')
    solver = FastHeuristic(example.problem, limit_shortest_path=3, seed=42).run()
    assert solver.solution is not None
    result = format_result(example, solver.solution,
                           solver.solving_instance.compute_nth_shortest_path,
                           checkpoint='reference', steps=0, seed=42)
    assert result['feasible']
    assert result['total_served_mbps'] >= sum(example.problem.data.min_flows)
    assert len(result['placements']) == 2
    assert len(result['demands']) == 4
    assert len(result['directed_links']) == 14
    assert all(row['latency_ms'] <= row['latency_budget_ms'] + 1e-6 for row in result['demands'])


def test_mismatched_node_order_fails(tmp_path):
    import json
    item = json.loads((ROOT / 'instances/train/factory_00.json').read_text())
    item['nodes'][0], item['nodes'][1] = item['nodes'][1], item['nodes'][0]
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(item))
    baseline = load_instance(ROOT / 'instances/train/factory_00.json')
    assert load_instance(path).signature != baseline.signature
    with pytest.raises(ValueError, match='Incompatible'):
        validate_compatible([load_instance(path)], baseline.signature)


def test_negative_capacity_fails(tmp_path):
    import json
    item = json.loads((ROOT / 'instances/train/factory_00.json').read_text())
    item['links'][0]['capacity_mbps'] = -1
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(item))
    with pytest.raises(ValueError, match='capacity_mbps'):
        load_instance(path)
