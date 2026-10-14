# Temporal orchestration: model, protocol, and limitations

This module is a **synthetic research simulator**, not an operational network
controller. The fixed-topology static problem (`Problem`, `assess`,
`feasible_seed`, `greedy_uplift`) is reused exactly. It is important not to
confuse the static PPO editing process with **physical time**.

## State and dynamics

A `TemporalScenario` provides realized integer demand ceilings `realized_mbps[t,d]`
and a **published exogenous forecast** `forecast_mbps[t,d]`. The generator builds
this forecast from the deterministic daily trend **before** adding stochastic,
autocorrelated innovations to realized traffic. The known time-dependent pattern
is an assumption of the toy workload; this is not a trained forecasting model.
At time `t`, an MPC optimizer sees actual demand and capacity at `t`, forecast
demand for `t+1,...`, and **only the current observed capacity status**, held
constant when planning the future. Future actual demands and capacities are
never read. PPO receives the same contemporary signals and the next four
published forecast steps.

A decision consists of service placements `p[t]`, demand route indices `r[t]`,
and feasible integer bandwidth allocations `q[t]`. For each physical time step,
objective terms are:

```
stage_reward = sum(q[t]) - migration_cost * changed_service_nodes
                         - route_change_cost * changed_demand_routes
                         - active_node_cost * active_compute_nodes
```

CPU, directed-link bandwidth and load-dependent latency constraints remain
hard. The discounted return uses the configured `discount`, by default 0.98.
The configuration is a **unit-normalized toy objective**, not empirical energy
consumption or validated migration expenses. Migration is not modeled with
transfer times, downtime or real-time traffic guarantees.

## Planning algorithms

- `keep`: preserve committed placement, routes and bandwidth (clamp to new
  contract limits and decrease flows only when needed to restore feasibility).
- `greedy`: preserve placement/routes, then greedily refill available bandwidth.
- `myopic` / `mpc1`: one-step optimization of immediate reward over a bounded
  set of feasible route/placement changes.
- `mpc4`, `mpc8`: receding-horizon **beam search over candidate allocations**,
  with a configurable candidate cap, beam width, and deterministic expansion
  budget. An optional wall-time cutoff protects online inference, but changes
  the realized search depth according to machine speed. **Not an exact MPC optimum.**
  `horizon=1` is identical to the myopic method by implementation.
- `ppo`: optional one `MultiDiscrete` placement/routing choice per *physical*
  period. The bandwidth allocator fills the resulting configuration; infeasible
  actions are explicitly rejected with a fallback and learning penalty.
- `auto` (online): run both MPC and PPO if available and select the higher
  *validated immediate* stage reward. This does not guarantee higher trajectory
  return, and never credits PPO when MPC wins or ties.

`candidate_allocations()` includes current placement/routes, an initial seed,
a strong routing-lookahead allocation, one-service relocations and single-demand
reroutes. Its search is bounded for a predictable computation budget. A planner
failure uses an explicit feasible fallback, never an unconstrained projection.

## Experiments and statistical discipline

`instances/temporal/` contains **12 train**, **4 validation**, and **4 held-out
test** scenarios, each 24 periods. Test includes both ordinary and stressful
unseen stochastic load patterns. Seeds, splits and generator logic are tracked.
Scenario JSON contains the full realized trajectory for offline evaluation only.

Train 5 (then 10) completely independent policies, e.g. 100k/500k/1M steps,
never tune on the test split, then compare against *keep, myopic, MPC4, MPC8*
on exactly the same realized test trajectories. Compare total undiscounted and
discounted return, Mbps-periods served, migrations, route changes, invalid
proposals/fallbacks, and CPU inference wall-time p50/p95. Checkpoint loading
is excluded from recurrent inference timing and should be reported separately.
For uncertainty, resample over independent training seeds and held-out scenario
units rather than treating many evaluation timesteps as independent trials.

**There are no PPO or RL-vs-MPC wins established by this version.** Gymnasium and
Stable-Baselines3 are optional; absence of these dependencies skips only the RL
tests and makes PPO scripts unavailable. No empirical comparisons should be
claimed before independently running actual PPO trainings.

## Causal online API

The HTTP interface takes a **snapshot** and returns exactly one validated
allocation. It does *not* accept the simulator's `realized_mbps` future matrix.
Example: `instances/temporal/online_example.json`.

Snapshot fields:

- `schema_version: 1`, `id`, and the same `base_instance` static schema;
- `observed_demand_mbps: [D]` (current actual demand upper bounds);
- `observed_capacity_scale: [L]` (current link capacity factors);
- `forecast_mbps: [K,D]` (published exogenous forecast; row 0 may be now);
- `absolute_period`, `episode_length` (for PPO phase observation);
- `previous: {placement:[S], routes:[D], quantities:[D]}` or null;
- `migration_cost`, `route_change_cost`, `active_node_cost`, `discount`.

It returns `selected_method`, `feasible`, `decision`, `next_previous`,
`candidate_stage_rewards`, `fallback`, `warnings`, and runtime. Clients must
persist and resend `next_previous` to account for temporal switching costs.
The snapshot converter fills internal future *realizations* with forecasts only
so the offline simulator data structure can be reused; no future observations
are included in the HTTP request. All exported decisions are rechecked with
`assess` under **contemporaneous actual constraints**.

The HTTP reference adapters have no authentication, TLS, rate limiting or
request-level idempotency; put them behind a configured trusted gateway and
use only locally trusted SB3 checkpoints (pickle risk). The model and planning
configuration are fixed on server startup, never selected by HTTP clients.
