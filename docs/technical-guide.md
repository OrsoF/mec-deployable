# MEC Deployable — Static engine and integration guide (v0.3)

This guide covers the retained static engine. See [Temporal design](temporal-design.md) for the v0.3 temporal model, one-decision-per-period PPO, switching costs, causal beam MPC, and snapshot interface. The [README figures](figures/README.md) describe the verified examples and their source data.

## Model and feasibility

A problem contains nodes with available vCPU, services with CPU requirements, directed arcs from undirected physical links (two independent directional capacities), and demands with integer Mbps bounds and maximum delay. A solution chooses (1) one node per service, (2) one of `K` deterministic candidate paths per demand, and (3) integer Mbps per demand.

For demand `i`, arc incidence `A[i,e]` and allocated Mbps `q[i]`, arc load is `L[e] = sum_i A[i,e] * q[i]`. Constraints: CPU load <= node capacity, arc load <= directional capacity, lower <= q <= upper, and demand latency `sum_e A[i,e] * (base[e] + slope[e]*L[e]) <= latency_budget[i]`. Every deployed decision is checked with `constraints.assess`; invalid edits are rejected. This is a **simplified deterministic model**.

## Scenario split

Run `python scripts/generate_instances.py --output instances`. Ten training examples vary crossed-link capacity (1–4 Mbps), local capacity (12–16 Mbps), core-link capacity (18–21 Mbps), maximum demand (5–8 Mbps), and latency (15–20 ms). The held-out test uses cross=2, local=14, core=20, max demand=7 Mbps, latency=25 ms. This is parameterized interpolation/extrapolation over **one topology**, with no measured industrial data.

On the held-out instance, `greedy` gets 18 Mbps, while `lookahead` and `multistart` reach 28 Mbps, matching the sum of all maximum demands and therefore proving optimality *for that instance*. Other instances do not automatically have certificates just because one heuristic wins.

## Algorithms

- `greedy`: build a feasible minimum-Mbps placement/routing, then repeatedly increment a feasible demand's Mbps. **Placement and route are fixed**.
- `lookahead`: start from greedy, try each single-demand reroute that preserves feasibility, then apply greedy uplift; accept strictly improving proposals until local convergence or the round limit.
- `multistart`: run lookahead from bounded feasible variations of a minimum-flow route vector. Neither search heuristic is globally optimal in general.
- `ppo`: Stable-Baselines3 MlpPolicy trained on the Gymnasium environment. Atomic edits change placement, route, or Mbps by one. `hybrid` adds route-and-uplift macro-actions (one admissible reroute followed by greedy increase); PPO gets no infeasible intermediate states.
- `auto`: selects the highest feasible score among lookahead and optionally PPO. **Labels the true selected method**, records failures, uses no false PPO attribution.
- `exact_small`: optional SciPy MILP on every enumerated placement and route combination. Fixed-route load and congestion latency become linear in Mbps. Search is exponential; only a completed enumeration or attainment of the demand upper bound certifies the optimum.

Reward is the change in served Mbps and a small no-op/infeasibility penalty. Static PPO operates on **per-instance optimization episodes**; external demand does not evolve during these episodes. The separate temporal module implements evolving traffic, migration/routing costs and a bounded MPC comparator. Observation normalization is implemented and frozen for inference. Future work includes action masking, topology generalization and calibrated uncertainty.

## Train, evaluate, benchmark

```bash
pip install -e '.[rl,test]'
pytest -q
python train.py --instances instances/train --output outputs/ppo_bottleneck --timesteps 100000 --action-mode hybrid --seed 42
python eval.py --checkpoint outputs/ppo_bottleneck --instance instances/eval/bottleneck_unseen.json --episodes 40 --stochastic
python benchmark.py --instances instances/eval --checkpoint outputs/ppo_bottleneck --episodes 40 --stochastic --output outputs/benchmark_ppo
```

`train.py` checks 10 file count and topology signature; saves policy zip, normalization and metadata including instance hashes, library versions and action mode. The policy is never trained on the test instance. `eval.py` evaluates **PPO itself** and shows both weak and stronger heuristic references. Multiple episodes and stochastic action sampling can improve *best-of-budget* score, but are not single-pass deterministic inference. Compare compute/time budgets fairly.

`benchmark.py` emits `results.csv` (one method, instance, seed, score and runtime per row) and `summary.json` (mean, standard deviation, runtime). Baseline methods are deterministic here so they are executed once per case. Benchmark runs include seed generation, model inference and final validation; model loading is excluded. It does not produce confidence bounds on out-of-sample performance and does not enforce equal number of solver operations.

## Input and output contract

`POST /decide` consumes the same `schema_version=1` JSON as `instances/eval/bottleneck_unseen.json`; keys are `id`, `units`, `nodes`, `links`, `services` and their `demands`. Units must be exactly `Mbps`, `ms`, `vCPU`. Invalid, nonfinite, fractional-Mbps and duplicate identifiers are rejected.

`decision_json` produces: `schema_version`, `instance`, UTC evaluation time, seed, feasible, objective, placements, routes, allocated quantity, measured latency, CPU loads, directional link loads. `DecisionEngine` additionally provides: `selected_method`, `requested_strategy`, `candidate_scores_mbps`, `candidate_run_ms`, `decision_wall_ms`, `warnings`, `demand_upper_bound_mbps` and `certified_optimal`. A `certified_optimal=true` flag is used only if total served Mbps equals the trivial demand upper bound, not on an unverified heuristic improvement.

```bash
python deploy.py --instance instances/eval/bottleneck_unseen.json --strategy lookahead --output outputs/decision.json
python serve.py --host 127.0.0.1 --port 8080 --strategy lookahead
curl -X POST http://127.0.0.1:8080/decide -H 'Content-Type: application/json' --data-binary @instances/eval/bottleneck_unseen.json
```

API: `GET /health` returns process health; `GET /ready` gives selected strategy and model-loaded status; `POST /decide` returns a validated decision or a JSON error (400 malformed input, 415 wrong MIME type, 413 oversized request, 422 rejected decision, 503 infeasibility). Request limit: 1 MiB. Server is synchronous in inference, with a lock around the policy RNG/state; deploy behind authentication, TLS and rate limits. No uploaded checkpoint or dynamic shell command is accepted. Stable-Baselines3 model formats involve pickle: accept checkpoints only from trusted authors.

## Deployment

```bash
docker build -t mec-deployable .
docker run --rm -p 127.0.0.1:8080:8080 mec-deployable
```

To run the PPO model, build `--build-arg INSTALL_RL=true` and mount a **trusted** local checkpoint at `/app/outputs` read-only. Running `train.py` inside the RL Docker image is possible only if output directory is writable and the required scripts/data are present; saving a trained checkpoint is not performed by default. Container includes one application worker and does not include a reverse proxy.

```bash
docker build --build-arg INSTALL_RL=true -t mec-deployable-rl .
docker run --rm -p 127.0.0.1:8080:8080 -v "$(pwd)/outputs:/app/outputs:ro" mec-deployable-rl python serve.py --host 0.0.0.0 --strategy auto --checkpoint outputs/ppo_bottleneck
```

The mount syntax above uses a POSIX shell; use an absolute host path for Windows Docker. Docker build/runtime is not verified in the recorded local audit. The CLI accepts only a locally configured checkpoint; HTTP clients cannot select or upload one.

Old figures/notebook were produced on the original less congested samples retained under `instances/legacy/`; their provenance is not a benchmark result for the new scenarios.
