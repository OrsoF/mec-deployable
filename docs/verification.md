# Local verification — 14 October 2026

Environment: Python 3.13, NumPy 2.3.5, NetworkX 3.6.1, SciPy 1.17.0, pytest 9.0.2.

Executed from repository root with `PYTHONPATH=src`:

```text
pytest -q
37 passed, 1 skipped

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
