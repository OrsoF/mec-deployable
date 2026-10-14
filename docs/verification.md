# Local verification — 14 October 2026

Environment: Python 3.13, NumPy 2.3.5, NetworkX 3.6.1, SciPy 1.17.0, pytest 9.0.2.

Executed from repository root with `PYTHONPATH=src`:

```text
pytest -q
46 passed, 2 skipped

python deploy.py --strategy greedy
routing_bottleneck_heldout: greedy -> 18/28 Mbps, feasible=True

python deploy.py --strategy lookahead
routing_bottleneck_heldout: lookahead -> 28/28 Mbps, feasible=True

python benchmark.py --instances instances/eval
Greedy 18 Mbps; routing lookahead 28 Mbps; multistart 28 Mbps.
```

Other exercised paths: deterministic instance generator reproduces committed JSON byte-for-byte; one-shot deployment creates validated JSON; local HTTP service handles readiness, valid inference and invalid requests; exact solver certifies an additional one-demand MILP problem.

**Not verified in this environment:** actual Stable-Baselines3 PPO training/inference (Gymnasium and stable-baselines3 absent), Docker build/runtime (Docker unavailable), public API security hardening, production latency, performance under realistic network traffic. The separate CI `rl-smoke` job installs optional RL dependencies and runs PPO model training/evaluation if GitHub Actions is configured.

**Benchmark interpretation:** The lookahead algorithm reaches the global demand upper bound on the reserved case and is therefore optimal. The repository does not currently provide evidence of PPO superiority over lookahead, let alone over general exact/MPC methods. Previous independent PyTorch PPO explorations are not packaged training results.

## v0.3 temporal additions

New tests cover deterministic temporal workload generation, split independence,
causal forecast availability, future-realization non-leakage, migration/routing
penalties, hard-feasibility preservation across periods, myopic/MPC1 equality,
stateful one-step deployment and HTTP integration. The optional Gymnasium
compliance test skips locally because Gymnasium is not installed.

A verified two-scenario time-independent **expansion-capped** comparison,
with `--max-cases 2 --max-candidates 5 --max-expansions 30 --horizons 1 4 8`,
gave mean cumulative rewards: keep 91.20, greedy 320.20, myopic 473.10,
MPC4 473.10 and MPC8 473.10. These MPC runs typically reached only ~3
steps because of the deterministic expansion budget; `mpc8` is the REQUESTED
horizon, not necessarily the COMPLETED horizon. Run with
`--max-expansions 0 --max-planning-ms 0` to attempt untruncated MPC, at
considerably higher computational cost. This run shows no advantage from MPC
on those two cases. These figures are **synthetic** and not deployed-network
measurements.

No PPO model was trained or evaluated in this runtime, because both
`gymnasium` and `stable_baselines3` are absent and network installation failed.
The CI `rl-smoke` job and the multi-seed training scripts are written but
await execution on a machine with these dependencies. Docker was not tested.
