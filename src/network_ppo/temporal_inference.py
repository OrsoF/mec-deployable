"""Single-period online decision adapter, with provenance and feasible fallback."""
from __future__ import annotations
from time import perf_counter
from .dynamics import parse_online_snapshot, solution_to_state
from .paths import PathCatalog
from .temporal_optimization import TemporalPlanner, initial_allocation
from .temporal_costs import stage_score
from .constraints import assess
from .solution_io import decision_json


class TemporalDecisionEngine:
    def __init__(self, *, strategy="mpc", horizon=4, beam_width=2,
                 max_candidates=9, checkpoint=None, seed=123, max_planning_ms=400):
        if strategy not in ("keep","greedy","myopic","mpc","ppo","auto"):
            raise ValueError("Unsupported temporal strategy")
        if strategy=="ppo" and checkpoint is None:
            raise ValueError("PPO requires checkpoint")
        self.strategy=strategy
        self.seed=seed
        self.horizon=horizon
        self.beam_width=beam_width
        self.max_candidates=max_candidates
        self.max_planning_ms=max_planning_ms
        self.policy=None
        self.model_warning=None
        if checkpoint:
            try:
                from .temporal_policy import TemporalPolicy
                self.policy=TemporalPolicy(checkpoint)
            except (OSError, ValueError, RuntimeError, ModuleNotFoundError, ImportError) as exc:
                if strategy != "auto":
                    raise
                self.model_warning=f"PPO unavailable: {type(exc).__name__}: {exc}"

    def decide(self, snapshot:dict) -> dict:
        scenario,previous=parse_online_snapshot(snapshot)
        if scenario.initial_time > 0 and previous is None:
            raise ValueError("previous decision is mandatory after the initial period")
        previous=previous if previous is not None else initial_allocation(scenario)
        t0=perf_counter()
        p=scenario.problem_at(0)
        catalog=PathCatalog(p,3)
        candidates=[]
        warnings=[]
        if self.model_warning:
            warnings.append(self.model_warning)
        if self.strategy not in ("ppo",):
            method="mpc" if self.strategy=="auto" else self.strategy
            planned=TemporalPlanner(strategy=method,horizon=self.horizon,
                                    beam_width=self.beam_width,max_candidates=self.max_candidates,
                                    max_planning_ms=self.max_planning_ms).decide(scenario,0,previous)
            score=stage_score(scenario,p,previous,planned.solution,catalog)
            candidates.append((method,planned.solution,score,planned.runtime_ms,planned.fallback))
        if self.strategy in ("ppo","auto") and self.policy is not None:
            try:
                sol,score,fallback,ms=self.policy.predict_one(scenario,previous,seed=self.seed)
                candidates.append(("ppo",sol,score,ms,fallback))
            except (ValueError,RuntimeError) as exc:
                if self.strategy=="ppo":
                    raise
                warnings.append(f"PPO rejected: {type(exc).__name__}: {exc}")
        if not candidates:
            raise RuntimeError("No temporal decision available")
        # Never credit PPO when the baseline wins a tie.
        selected=max(candidates,key=lambda row:row[2].reward)
        method,solution,score,ms,failed=selected
        check=assess(p,solution,catalog)
        if not check.feasible:
            raise RuntimeError("Unsafe decision not exported")
        return {"schema_version":1,"scenario":scenario.id,
                "absolute_period":scenario.initial_time,"selected_method":method,
                "requested_strategy":self.strategy,"feasible":True,
                "decision":solution_to_state(solution),
                "allocation":decision_json(p,solution,catalog,checkpoint="temporal:"+method,
                                           policy_steps=1 if method=="ppo" else 0,seed=self.seed),
                "score":score.__dict__,
                "fallback":failed,"warnings":warnings,
                "planning_horizon_reached":planned.lookahead_evaluated if method!="ppo" else None,
                "planning_budget_exhausted":planned.budget_exhausted if method!="ppo" else None,
                "candidate_stage_rewards":{name:metric.reward for name,_,metric,_,_ in candidates},
                "runtime_ms":round((perf_counter()-t0)*1000,4),
                "next_previous":solution_to_state(solution)}
