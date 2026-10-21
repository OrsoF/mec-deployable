<h1 align="center">MEC Deployable</h1>

<p align="center">PPO-based service placement, routing, and bandwidth allocation for edge networks.</p>

<p align="center">
  <a href="https://github.com/OrsoF/mec-deployable/actions/workflows/ci.yml"><img src="https://github.com/OrsoF/mec-deployable/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-2563a6" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-087f82" alt="MIT license"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="docs/technical-guide.md">Documentation</a>
</p>

Where should a service run, and how should traffic reach it? Edge nodes have limited computing resources, network links have limited bandwidth, and applications have latency requirements. **MEC Deployable uses Proximal Policy Optimization (PPO)** to learn how to place services, route traffic, and allocate bandwidth within these constraints.

<picture>
  <source media="(max-width: 600px)" srcset="docs/assets/ppo-placement-mobile.svg">
  <img src="docs/assets/ppo-placement.svg" alt="Offline PPO training produces a saved policy. During inference, the policy guides a sequence of validated edits before JSON export. An illustrative allocation shows video analytics on the north edge and telemetry on the south edge, with two demand routes highlighted." width="1040">
</picture>

*Repository topology with an illustrative, validated allocation; this is not a PPO evaluation. Two of four demand routes are highlighted.*

## How it works

1. **Train offline.** PPO learns from network scenarios in a Gymnasium environment.
2. **Generate a decision.** A saved policy guides placement, routing, and bandwidth edits from the current network state.
3. **Validate and export.** Check CPU capacity, directed bandwidth, integer traffic bounds, and load-dependent latency; return a feasible decision as JSON.

Train offline, then reuse the saved policy through the CLI or HTTP API. Static inference runs a sequence of validated edits and returns the best feasible state visited.

## Quick start

Python **3.10+**, from the repository root. Install the PPO dependencies, train a policy, then evaluate it on the held-out instance:

### Install

```bash
python -m pip install -e '.[rl]'
```

### Train and evaluate

```bash
python train.py --timesteps 100000 --output outputs/ppo_bottleneck
python eval.py --checkpoint outputs/ppo_bottleneck --output outputs/ppo_decision.json
```

### Serve with Docker

Serve the trained policy through the HTTP API:

```bash
docker build --build-arg INSTALL_RL=true -t mec-deployable .
docker run --rm -p 127.0.0.1:8080:8080 --mount "type=bind,source=$PWD/outputs,target=/app/outputs,readonly" mec-deployable python serve.py --host 0.0.0.0 --strategy ppo --checkpoint outputs/ppo_bottleneck
```

`POST /decide` accepts a network instance and returns placements, routes, allocated Mbps, and validation diagnostics. [Input format and deployment details](docs/technical-guide.md#input-and-output-contract).

**Measured execution time.** A local CPU run recorded **42.6 ms median / 59.4 ms p95** for a complete validated decision: one 96-edit episode, model already loaded, 100 requests on one synthetic instance, and a 512-step smoke checkpoint. This measures execution speed, not placement quality. [Timed scope, hardware, checkpoint, and reproduction](docs/ppo-latency.md).

## Scope and documentation

A research and engineering demonstrator using synthetic scenarios. No trained model is bundled and no network changes are applied. Policies require the same ordered topology and service/demand identities. Feasibility applies to the simplified network model. Load only trusted checkpoints; the HTTP adapter needs a gateway providing authentication and TLS. Docker runtime remains unverified locally.

[Technical guide](docs/technical-guide.md) · [Temporal control](docs/temporal-design.md) · [Tests and CI](docs/ci-verification.md) · [Figure sources](docs/figures/README.md) · [MIT license](LICENSE)

## Related Projects

- [mec-placement-benchmarks](https://github.com/OrsoF/mec-placement-benchmarks) — Benchmarking optimization and learning methods for constrained edge-network orchestration.
- [mdpforge](https://github.com/OrsoF/mdpforge) — A Python framework for finite MDP modeling, solving, and benchmarking.

More about my research, publications, and software projects: [orsof.github.io](https://orsof.github.io/).
