# PPO industrial MEC orchestration — MVP

Two CLI entry points built on the MIT-licensed **mecbench** constraint engine
(from the provided `mec-placement-benchmarks` archive):

```bash
python -m pip install -e .
python train.py                         # trains on exactly 10 JSON instances
python eval.py                          # evaluates an unseen example instance
```

A short CPU smoke experiment can use `python train.py --timesteps 1024` (not
meaningful learning); a real experiment should use substantially more training
steps, multiple random seeds and evaluate against the heuristic baseline.

The trained model is written to `outputs/ppo_factory/` (model.zip,
vecnormalize.pkl, metadata.json). The evaluation decision is written to
`outputs/decision.json`, with validated service placements, concrete routes,
allocated Mbps, per-demand latencies, CPU utilization and directed link loads.
It also reports the deterministic `fast_heuristic` reference score (no silent fallback).

## Input file

```bash
python eval.py --checkpoint outputs/ppo_factory \
  --instance instances/eval/factory_shift_unseen.json \
  --output outputs/decision.json --episodes 1
```

A standalone JSON describes:

- `schema_version: 1`, an `id`, and units `flow=Mbps`, `latency=ms`, `compute=vCPU`;
- `nodes`: `id`, available `cpu` (vCPU);
- `links`: `source`, `target`, `capacity_mbps`, `latency_base_ms`,
  `latency_per_mbps_ms`;
- `services`: `id`, required `cpu`, and `demands` with `id`, `origin`,
  `min_mbps`, `max_mbps`, `max_latency_ms`.

Links are **bidirectional** and capacities apply to each directed arc (they
are not shared across directions); total arc latency is modeled by
`latency_base_ms + latency_per_mbps_ms * total_arc_mbps`. Each service gets
**exactly one placement**. Each demand picks one of the three shortest simple
routes and an **integer** number of Mbps (increments of 1 Mbps).

The JSON in `instances/train/` holds exactly 10 synthetic examples with the same
node/edge ordering, service ordering and demand ordering, but changing traffic
and some capacities. `instances/eval/factory_shift_unseen.json` is held out.

### Critical V1 restriction: fixed indexed network

PPO with an MLP and MultiDiscrete actions is dimensionally fixed. Evaluation
must have identical node IDs, directed arc ordering, service IDs and demand IDs
as training (checked against `metadata.json`). Values such as capacities,
demand max/min, latencies and demand origins may change. A new arbitrary
industrial network topology needs a mapping/padding/GNN design and **retraining**;
this MVP explicitly rejects incompatible input instead of silently producing
an invalid decision.

The default evaluation uses deterministic PPO actions (one episode). For best-of-5
stochastic policy rollouts, use `--episodes 5 --stochastic`.

The extra observation includes demand origins and link delay coefficients. A
per-instance feasible seed is initialized by `fast_heuristic` and PPO improves
from minimum flow. Rewards follow changes in feasible total Mbps; invalid
moves are rejected. `VecNormalize` parameters are saved and reloaded for
inference. SB3 PPO timesteps may exceed the requested timesteps due to full
rollout collection; the actual count is in `metadata.json`.

### Model accuracy and limitations

The included site is **synthetic** and illustrative, not an operator-certified
industrial network. This proof of concept optimizes *sum of allocated demand
Mbps* subject to capacity, CPU, and simplified latency constraints. It does not
model availability, replicas, jitter, queueing, migration cost, tenant isolation,
SLA weights or multi-path splitting. It is not intended for automated production
network changes. Always inspect `decision.json` and compare to deterministic
`fast_heuristic` or an MILP baseline before a human approves an action.

Because `fast_heuristic` seeds PPO with placements (but only minimum volumes),
results combine deterministic initialization with RL refinement, rather than
isolating the PPO contribution. Report both performance and latency against
that baseline in subsequent benchmarking. Generalization from ten scenarios
should be treated as a small research experiment, not an industrial guarantee.

## Tests

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

## Code provenance

The `src/mecbench` namespace is copied from the user's provided source
archive. The new `src/mec_ppo` package plus the two entry points provide the
training/evaluation interface and JSON-to-domain adapters. Keep the included
`LICENSE` (MIT) when redistributing this code.

## Docker (CPU)

```bash
docker build -t mec-network-ppo .
docker run --rm -v "$(pwd)/outputs:/app/outputs" mec-network-ppo python train.py --timesteps 100000
docker run --rm -v "$(pwd)/outputs:/app/outputs" mec-network-ppo python eval.py
```

For an external JSON, mount its directory read-only as `/app/external` and pass
`--instance /app/external/your_instance.json`. The image installs PyTorch and
Stable-Baselines3, which can be relatively large. CPU inference is supported;
production services would usually prebuild and scan an image with pinned
versions rather than build from floating dependency ranges.
