"""Cached and validated temporal PPO inference (trusted checkpoints only)."""
from __future__ import annotations
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from .problem import validate_compatible
from .dynamics import TemporalScenario
from .temporal_env import TemporalNetworkEnv


class TemporalPolicy:
    def __init__(self, checkpoint: str | Path):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        self.directory=Path(checkpoint)
        self.metadata=json.loads((self.directory/"metadata.json").read_text(encoding="utf-8"))
        if self.metadata.get("schema_version") != 3 or self.metadata.get("mode") != "temporal":
            raise ValueError("Checkpoint is not a temporal PPO model (schema version 3)")
        self.model=PPO.load(str(self.directory/"model.zip"),device="cpu")
        self.DummyVecEnv=DummyVecEnv
        self.VecNormalize=VecNormalize

    def rollout(self, scenario: TemporalScenario, *, deterministic: bool = True,
                seed: int = 123) -> list[dict]:
        """Metrics from a COMPLETE policy rollout; no heuristic best-of-selection."""
        validate_compatible([scenario.base],reference=self.metadata["signature"])
        n_paths=int(self.metadata["paths"])
        steps=int(self.metadata["forecast_steps"])
        instance=TemporalNetworkEnv(scenario,paths=n_paths,forecast_steps=steps)
        vec=self.DummyVecEnv([lambda:instance])
        normalized=self.VecNormalize.load(str(self.directory/"vecnormalize.pkl"),vec)
        normalized.training=False
        normalized.norm_reward=False
        self.model.set_random_seed(seed)
        normalized.seed(seed)
        records=[]
        try:
            obs=normalized.reset()
            for t in range(scenario.horizon):
                start=perf_counter()
                action,_=self.model.predict(obs,deterministic=deterministic)
                obs,reward,done,infos=normalized.step(action)
                ms=(perf_counter()-start)*1000
                info=infos[0]
                records.append({"period":t,"strategy":"ppo","reward":float(reward[0]),
                                "served_mbps":info["served_mbps"],
                                "migrations":info["migrations"],"route_changes":info["route_changes"],
                                "active_nodes":info["active_nodes"],
                                "fallback":bool(info["fallback"]),"runtime_ms":ms,
                                "valid_proposal":bool(info["valid_proposal"])})
                if done[0] and t != scenario.horizon-1:
                    raise RuntimeError("Temporal PPO episode ended earlier than expected")
        finally:
            normalized.close()
        return records

    def predict_one(self, scenario: TemporalScenario, previous, *, t: int = 0,
                    deterministic: bool = True, seed: int = 123):
        """Online inference on a snapshot, with an explicit fallback indicator."""
        from .temporal_optimization import allocate, TemporalPlanner
        from .temporal_costs import stage_score
        from .paths import PathCatalog
        validate_compatible([scenario.base], reference=self.metadata["signature"])
        paths=int(self.metadata["paths"])
        env=TemporalNetworkEnv(scenario, paths=paths,
                               forecast_steps=int(self.metadata["forecast_steps"]))
        env.t=t
        env.previous=previous.copy()
        # Freeze training observation normalization. The normalizer is cached
        # on this policy runtime and reused by the HTTP API.
        if not hasattr(self,"_normalizer"):
            self._normalizer=self.VecNormalize.load(
                str(self.directory/"vecnormalize.pkl"),self.DummyVecEnv([lambda:env]))
            self._normalizer.training=False
            self._normalizer.norm_reward=False
        obs=self._normalizer.normalize_obs(env._observation()[None,:])
        self.model.set_random_seed(seed)
        begin=perf_counter()
        action,_=self.model.predict(obs,deterministic=deterministic)
        p=scenario.problem_at(t)
        catalog=PathCatalog(p,paths)
        s=len(p.services)
        chosen=allocate(p,catalog,np.asarray(action[0][:s],dtype=int),
                        np.asarray(action[0][s:],dtype=int))
        fallback=chosen is None
        if fallback:
            chosen=TemporalPlanner(strategy="myopic",paths=paths).decide(scenario,t,previous).solution
        metrics=stage_score(scenario,p,previous,chosen,catalog)
        return chosen, metrics, fallback, (perf_counter()-begin)*1000
