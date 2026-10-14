# Change log

## v0.3 — October 2026

- Preserved v0.2 static network constraint model, JSON API and PPO edit environment.
- Added 12 training, 4 validation and 4 independent test dynamic trajectories.
- Added causal public traffic forecasts, contemporaneous link-capacity observations,
  and switching/active-node costs per physical period.
- Added deterministic-cap bounded beam-search MPC with optional wall-time cutoff;
  exposed completed lookahead depth and explicit fallback.
- Added Gymnasium `TemporalNetworkEnv` with one full allocation per time step.
- Added independent PPO training orchestration (`train_multiseed.py`), a temporal
  training script, static multi-checkpoint comparison and paired temporal benchmark.
- Added temporal one-step CLI and HTTP inference, causal snapshot schema,
  checkpoint provenance and full validated allocation output.
- Added temporal unit and integration tests, optional Gymnasium smoke tests, CI
  steps and new deployment/benchmark documentation.
- Locally verified: 46 tests passed, 2 optional RL groups skipped; static
  lookahead still serves 28/28 Mbps. No real SB3 training result is claimed.

See `docs/temporal-design.md` and `docs/verification.md` for methodology and
limitations. In particular, all traffic and cost coefficients are synthetic.
