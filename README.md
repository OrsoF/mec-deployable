# MEC Deployable — Constrained edge-network orchestration

[![CI](https://github.com/OrsoF/mec-deployable/actions/workflows/ci.yml/badge.svg)](https://github.com/OrsoF/mec-deployable/actions/workflows/ci.yml)

A small, auditable network-decision engine that combines **placement, routing and integer-bandwidth allocation** under CPU, directed-link capacity and load-dependent latency constraints. Includes reproducible congestion scenarios, transparent optimization baselines, optional **PPO**, validated JSON decisions, a CLI, a small HTTP adapter and Docker packaging.

**Status:** reference research/engineering demonstrator using **synthetic data**, not a production network controller. No model is bundled and no orchestrator actions are performed automatically.

## Why this benchmark is nontrivial

A fixed-route throughput greedy fails when cross-links are congested and detours require a route modification that does not immediately increase served traffic. On the held-out synthetic instance:

| Method | Served Mbps | Feasible | Optimality |
|---|---:|---|---|
| Fixed-route greedy | **18 / 28** | Yes | Suboptimal |
| Routing lookahead + greedy uplift | **28 / 28** | Yes | Optimal by the demand upper bound |
| Multistart routing lookahead | **28 / 28** | Yes | Optimal by the demand upper bound |
| PPO | **Not measured in this rewrite** | Enforced by validator | No guarantee |

These baseline results were reproduced by `pytest` and `benchmark.py`. PPO's performance must be measured **after actual Stable-Baselines3 training**; it is not expected to dominate the stronger baselines automatically. A separate exploratory PyTorch PPO run from an earlier experiment is not equivalent to the packaged Stable-Baselines3 implementation.

The 10 generated training cases vary link capacity, demand and latency budget without changing ordered topology, service or demand identities. A different combination of parameters is held out for evaluation. This controlled split is **not evidence of out-of-distribution generalization** or of general optimality.

## Quick start

Python 3.10+; NumPy and NetworkX are sufficient for the heuristic decision engine.

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python deploy.py --strategy greedy --output outputs/greedy.json
python deploy.py --strategy lookahead --output outputs/decision.json
python benchmark.py --instances instances/eval --output outputs/benchmark
```

**Train PPO (additional dependencies and compute required):**

```bash
python -m pip install -e '.[rl,test]'
python train.py --instances instances/train --output outputs/ppo_bottleneck \
  --timesteps 100000 --episode-steps 96 --paths 3 --seed 42 \
  --action-mode hybrid
python eval.py --instance instances/eval/bottleneck_unseen.json \
  --checkpoint outputs/ppo_bottleneck --episodes 40 --stochastic \
  --output outputs/ppo_decision.json
python benchmark.py --instances instances/eval --checkpoint outputs/ppo_bottleneck \
  --episodes 40 --stochastic --output outputs/benchmark_ppo
```

For a fast smoke test, replace `--timesteps 100000` with `--timesteps 512`; a smoke test does **not** establish policy performance. The training script writes `model.zip`, `vecnormalize.pkl` and `metadata.json`; evaluation checks the topology signature and uses frozen observation-normalization statistics.

**Production-style decision command:**

```bash
python deploy.py --strategy auto --checkpoint outputs/ppo_bottleneck \
  --instance instances/eval/bottleneck_unseen.json --output outputs/decision.json
```

In `auto`, the engine evaluates the lookahead baseline and, if a compatible checkpoint is available, PPO. It returns the **better feasible** candidate and records `selected_method`, candidate scores, runtimes and warnings. If PPO fails, it falls back to the baseline with an explicit warning. `--strategy ppo` has **no silent fallback**. Default deployment without a checkpoint uses `lookahead`.

## JSON inference and HTTP integration

Input: versioned instance JSON with fixed-topology node/link/service/demand identifiers, CPU capacity, bandwidth, latency and demand bounds. Output: placements, directed routes, allocated Mbps, measured latencies and loads, feasibility, algorithm provenance and diagnostics. No network changes are applied by the demo.

```bash
python serve.py --host 127.0.0.1 --port 8080 --strategy lookahead
curl -s http://127.0.0.1:8080/ready
curl -s -X POST http://127.0.0.1:8080/decide \
  -H 'Content-Type: application/json' \
  --data-binary @instances/eval/bottleneck_unseen.json
```

The HTTP service caches a trusted PPO model when configured, bounds request bodies to 1 MiB, validates all inputs, serializes accesses to model state and checks constraints before output. **It does not provide authentication or TLS**, and should only be deployed behind a configured reverse proxy on a trusted network. Never load untrusted model checkpoints; Stable-Baselines3 artifacts use pickle internally.

Docker baseline API (no PyTorch dependency):

```bash
docker build -t mec-deployable .
docker run --rm -p 127.0.0.1:8080:8080 mec-deployable
```

Docker with PPO dependencies and a **locally trusted** checkpoint:

```bash
docker build --build-arg INSTALL_RL=true -t mec-deployable-rl .
docker run --rm -p 127.0.0.1:8080:8080 \
  -v "$(pwd)/outputs:/app/outputs:ro" mec-deployable-rl \
  python serve.py --host 0.0.0.0 --strategy auto \
  --checkpoint outputs/ppo_bottleneck
```

## Architecture

```text
mec-deployable/
├── src/network_ppo/
│   ├── problem.py            # domain types, compatible topology signature
│   ├── instance_io.py        # strict parser for JSON files and HTTP bodies
│   ├── paths.py              # stable candidate routes
│   ├── constraints.py        # independent feasibility checker
│   ├── heuristic.py          # feasible seed, greedy, routing search, multistart
│   ├── exact.py              # tiny-instance MILP enumeration / certificates
│   ├── environment.py        # optional Gymnasium PPO environment
│   ├── inference.py          # cached model, baseline selection, safe fallback
│   └── solution_io.py        # machine-readable verified decision
├── scripts/generate_instances.py
├── instances/templates/       # generator input
├── instances/train/           # 10 deterministic congestion scenarios
├── instances/eval/            # one held-out congestion scenario
├── instances/legacy/          # original uncongested sample scenarios
├── train.py                   # PPO model + normalization + metadata
├── eval.py                    # PPO-only held-out evaluation vs two baselines
├── benchmark.py               # CSV, JSON, multi-seed comparisons
├── deploy.py                  # one-shot JSON inference
├── serve.py                   # minimal POST /decide API
├── tests/                    # unit/integration/regression tests
├── .github/workflows/ci.yml  # core and PPO smoke tests
└── Dockerfile
```

**Regenerate the committed split:**

```bash
python scripts/generate_instances.py --output instances
```

**Optional exact solver** (`python -m pip install scipy`): `network_ppo.exact.solve_exact_small` enumerates placements and candidate routes and solves the resulting integer linear programs. Certificates are valid only if all configurations were searched or the global demand upper bound was reached. Time and configuration limits are explicit; this method is intended for small unit-test cases, not the deployment path.

## Engineering scope and limitations

- Directed capacities are independent per link direction. The latency model is additive with linear congestion cost; it does **not** model packet losses, probabilistic SLA satisfaction, traffic uncertainty or real deployment economics.
- The PPO environment represents *sequential edits to a static optimization instance*, not a time-evolving MDP of real traffic. Its `hybrid` mode adds rerouting-and-uplift macro-actions, which share a strong local-optimization primitive with the lookahead baseline. Compare them fairly.
- The chosen action space and model architecture assume the same ordered topology, links, service identities and demand identities at train and inference. New topologies require retraining or a graph-based architecture.
- PPO results should be reported over independent training seeds and held-out instances, with stochastic versus deterministic inference explicitly distinguished and latency budgets accounted for. The benchmark reports wall times, but **does not enforce matched algorithmic compute budgets**.
- The server is a reference integration adapter, not a hardened, horizontally scalable API. It does not implement access control, resource quotas, persistence or operational monitoring.
- The old notebook and its figures are retained for reference to the **original legacy factory scenarios**, not evidence about the new congested benchmark or PPO performance. See [`docs/technical-guide.md`](docs/technical-guide.md) for the exact mechanics and deployment contract.

License: MIT.
