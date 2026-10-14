"""Temporal decision invariants, causal forecasts, deployment, reproducibility."""
from __future__ import annotations
import copy
import json
import subprocess
import sys
from dataclasses import replace
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from network_ppo.constraints import assess
from network_ppo.dynamics import (load_temporal_scenario,parse_online_snapshot,
                                  parse_temporal_scenario,solution_to_state)
from network_ppo.paths import PathCatalog
from network_ppo.temporal_costs import stage_score
from network_ppo.temporal_inference import TemporalDecisionEngine
from network_ppo.temporal_optimization import TemporalPlanner,allocate,initial_allocation
from scripts.generate_temporal_cases import generate

ROOT=Path(__file__).resolve().parents[1]
TEST=ROOT/'instances/temporal/test/temporal_test_daily_900.json'
SNAPSHOT=ROOT/'instances/temporal/online_example.json'


def case():
    return load_temporal_scenario(TEST)


def test_temporal_split_and_roundtrip():
    s=case()
    assert s.split=='test' and s.horizon==24
    assert len(s.realized_mbps[0])==len(s.base.demands)
    assert s.problem_at(0).demands[0].max_mbps == int(s.realized_mbps[0,0])
    with pytest.raises(ValueError,match='minimum'):
        parse_temporal_scenario({**json.loads(TEST.read_text()),'realized_mbps':[[0]*4]*24})
    with pytest.raises(ValueError,match='schema_version'):
        parse_temporal_scenario({})


def test_generator_deterministic_and_disjoint_splits(tmp_path):
    generate(tmp_path)
    for file in (ROOT/'instances/temporal').glob('*/*.json'):
        counterpart=tmp_path/file.relative_to(ROOT/'instances/temporal')
        assert counterpart.read_bytes()==file.read_bytes()
    train={p.read_bytes() for p in (tmp_path/'train').glob('*.json')}
    test={p.read_bytes() for p in (tmp_path/'test').glob('*.json')}
    validation={p.read_bytes() for p in (tmp_path/'validation').glob('*.json')}
    assert len(train)==12 and len(validation)==4 and len(test)==4
    assert not (train&test or train&validation or validation&test)


def test_stage_costs_and_infeasible_rejection():
    sc=case()
    p=sc.problem_at(0)
    cat=PathCatalog(p,3)
    previous=initial_allocation(sc)
    chosen=allocate(p,cat,previous.placement,previous.routes)
    result=stage_score(sc,p,previous,chosen,cat)
    assert result.reward==pytest.approx(result.served_mbps-result.energy_cost)
    altered=chosen.copy()
    altered.quantities[0]=1000
    with pytest.raises(ValueError,match='Infeasible'):
        stage_score(sc,p,previous,altered,cat)
    switched=chosen.copy()
    switched.routes[0]=1
    if assess(p,switched,cat).feasible:
        score=stage_score(sc,p,previous,switched,cat)
        assert score.route_changes==1
        assert score.reward==pytest.approx(score.served_mbps-sc.route_change_cost-score.energy_cost)


def test_mpc_never_peeks_at_future_realizations():
    sc=case()
    previous=initial_allocation(sc)
    arrays=sc.realized_mbps.copy()
    arrays[1:,:]+=50  # would radically change an illegally clairvoyant MPC
    tampered=replace(sc,realized_mbps=arrays)
    planner=TemporalPlanner(strategy='mpc',horizon=4,beam_width=2,max_candidates=5)
    a=planner.decide(sc,0,previous)
    b=planner.decide(tampered,0,previous)
    assert solution_to_state(a.solution)==solution_to_state(b.solution)
    assert a.estimated_value==pytest.approx(b.estimated_value)


def test_mpc_horizon_one_matches_myopic():
    sc=case()
    old=initial_allocation(sc)
    a=TemporalPlanner(strategy='myopic',max_candidates=7).decide(sc,0,old)
    b=TemporalPlanner(strategy='mpc',horizon=1,max_candidates=7).decide(sc,0,old)
    assert solution_to_state(a.solution)==solution_to_state(b.solution)


def test_temporal_baselines_validate_every_period():
    sc=case()
    for strategy in ('keep','greedy','myopic','mpc'):
        previous=initial_allocation(sc)
        planner=TemporalPlanner(strategy=strategy,horizon=3,beam_width=2,max_candidates=5)
        for t in range(5):
            decision=planner.decide(sc,t,previous)
            p=sc.problem_at(t)
            assert assess(p,decision.solution,PathCatalog(p,3)).feasible
            _=stage_score(sc,p,previous,decision.solution,PathCatalog(p,3))
            previous=decision.solution


def test_online_snapshot_denies_realized_future():
    snap=json.loads(SNAPSHOT.read_text())
    assert 'realized_mbps' not in snap
    sc,prior=parse_online_snapshot(snap)
    assert prior is None and sc.initial_time==0
    assert np.array_equal(sc.realized_mbps[0],snap['observed_demand_mbps'])
    assert np.array_equal(sc.realized_mbps[1:],np.array(snap['forecast_mbps'][1:],dtype=int))
    future_leak=copy.deepcopy(snap)
    future_leak["realized_mbps"]=[[99]*4]
    with pytest.raises(ValueError,match="future realized"):
        parse_online_snapshot(future_leak)
    bad=copy.deepcopy(snap)
    bad['observed_demand_mbps']=[1]
    with pytest.raises(ValueError,match='dimensions'):
        parse_online_snapshot(bad)


def test_temporal_inference_returns_previous_and_provenance():
    raw=json.loads(SNAPSHOT.read_text())
    engine=TemporalDecisionEngine(strategy='auto',horizon=3,max_candidates=5)
    result=engine.decide(raw)
    assert result['feasible'] and result['selected_method']=='mpc'
    assert result['next_previous']==result['decision']
    assert result['allocation']['feasible']
    assert len(result['allocation']['directed_links'])==2*len(case().base.links)
    assert len(result['decision']['quantities'])==4
    raw['previous']=result['next_previous']
    result2=engine.decide(raw)
    assert result2['feasible']
    json.dumps(result2,allow_nan=False)


def test_temporal_cli_and_http(tmp_path):
    output=tmp_path/'decision.json'
    run=subprocess.run([sys.executable,'temporal_deploy.py','--snapshot',str(SNAPSHOT),
                        '--strategy','myopic','--max-candidates','5', '--output',str(output)],
                       cwd=ROOT,capture_output=True,text=True,timeout=60)
    assert run.returncode==0,run.stderr
    assert json.loads(output.read_text())['feasible']
    from serve_temporal import handler_factory
    server=ThreadingHTTPServer(('127.0.0.1',0),handler_factory(
        TemporalDecisionEngine(strategy='myopic',max_candidates=5)))
    thread=Thread(target=server.serve_forever,daemon=True)
    thread.start()
    url=f'http://127.0.0.1:{server.server_address[1]}'
    try:
        with urlopen(url+'/ready',timeout=3) as r:
            assert json.load(r)['status']=='ready'
        with urlopen(Request(url+'/decide',data=SNAPSHOT.read_bytes(),
                             headers={'Content-Type':'application/json'}),timeout=20) as r:
            assert json.load(r)['feasible']
        with pytest.raises(HTTPError) as e:
            urlopen(Request(url+'/decide',data=b'{}',
                            headers={'Content-Type':'application/json'}),timeout=3)
        assert e.value.code==400
    finally:
        server.shutdown();server.server_close();thread.join(timeout=3)
