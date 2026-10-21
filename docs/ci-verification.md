# CI verification

## Pre-push verification — 21 October 2026

The hosted [CI run for commit `49527bd`](https://github.com/OrsoF/mec-deployable/actions/runs/39578497165)
passed all three original jobs: `unit (3.10)`, `unit (3.12)`, and `rl-smoke`.
The additional local changes below have not been pushed or checked by a new hosted run.

Ruff was not part of the workflow. A new `lint` job installs the pinned
`.[lint]` extra (Ruff 0.17.1), runs `python -m ruff check .`, and checks formatting
with `python -m ruff format --check .`. Repository settings target Python 3.10
and check import order, Pyflakes errors, and E4/E7/E9 rules. Narrow E402
exceptions preserve notebook path setup and optional Gymnasium test skipping.
Outputs are excluded. Python and notebook source formatting, import cleanup,
local variable names, and a SHA helper were normalized without changing the
algorithms, parameters, error contracts, or recorded experimental results.

| Check on the updated working tree | Result |
| --- | --- |
| Full Python 3.12 suite with RL dependencies | 52 passed, no skips; two existing Gymnasium observation-bound warnings |
| Fresh Python 3.12 environment installed with `.[test,lint]`, without RL | 46 passed; the two optional RL modules skipped as intended |
| Dependency consistency in both environments | Passed |
| Ruff lint and format in both environments | Passed; 48 files already formatted |
| Static and temporal CLI decisions and CI benchmark commands | Passed |
| Static and temporal PPO training (512 steps), evaluation, and temporal RL benchmark | Passed |
| Wheel and source distribution | Built successfully |
| Workflow syntax and action inputs | actionlint 1.7.12 passed; official download SHA-256 verified |
| Figure generation | All 15 asset files deterministic across two runs; only source hashes in the provenance changed from the commit |
| Notebook and compatibility review | Outputs, metadata and calculation AST preserved; all 38 tracked Python files parse with Python 3.10 grammar |
| Git whitespace validation | `git diff --check` passed |

Local logs and generated artifacts are in ignored `outputs/prepush-verification/`.
To reproduce the new lint job after installation:

```bash
python -m pip install -e '.[lint]'
python -m ruff check .
python -m ruff format --check .
```

Local execution used Windows and Python 3.12.14; the new changes still need the
hosted Ubuntu matrix after push. Python 3.10 grammar validation is not a local
Python 3.10 runtime test. Docker is unavailable on this machine and remains
unverified. The existing setuptools license-table deprecation warning is
nonfatal. No changes were committed or pushed by this verification.

## Historical verification — 14 October 2026

The existing GitHub Actions workflow was inspected and every command in its three
jobs was reproduced in isolated local environments. These runs were on Windows;
the hosted workflow runs on Ubuntu and must be rerun after the fixes are pushed.

## Hosted failure and fixes

The [CI run on commit ee79378](https://github.com/OrsoF/mec-deployable/actions/runs/38581665477)
failed in both unit jobs and in static PPO training. `instances/train` contained
20 JSON files, while the code and tests require exactly 10 compatible scenarios.
The ten original `factory_*` training files had already been copied into
`instances/legacy/train` but were still present in the active training directory.

- Removed those redundant copies, plus the redundant legacy evaluation file,
  only after verifying byte-identical SHA-256 hashes against the retained legacy
  files. The active static dataset now contains 10 congestion scenarios and
  their held-out evaluation case.
- Updated the legacy notebook and figure provenance to use `instances/legacy/`.
  Updated its PPO comparison to read the current evaluation output field.
- Made both dataset generators write UTF-8 bytes with LF line endings. Their
  byte-for-byte reproduction tests now work on Windows as well as Linux.
- Included SciPy in the test extra so that the exact-solver tests run in CI.
  Previously, they were silently skipped in both unit environments.
- Added `pip check` to both CI jobs to catch incompatible installed dependencies.

## Reproduced jobs

| Job | Local environment | Result |
| --- | --- | --- |
| `unit (3.10)` | Python 3.10.22, NumPy 2.2.6, NetworkX 3.4.2, SciPy 1.15.3 | 46 tests passed; 2 optional RL modules skipped; every CLI and benchmark step passed |
| `unit (3.12)` | Python 3.12.14, NumPy 2.5.4, NetworkX 3.7, SciPy 1.18.1 | 46 tests passed; 2 optional RL modules skipped; every CLI and benchmark step passed |
| `rl-smoke` | Python 3.11.17, Gymnasium 1.4.0, SB3 2.9.0, PyTorch 2.14.1 CPU | 6 RL tests passed; both 512-step training runs and their evaluations passed |

The unit jobs intentionally omit RL dependencies. Their two skipped modules
are exercised by `rl-smoke`. All three environments pass `pip check`.
A full run with all optional test and RL dependencies on Python 3.12 passes
**52 tests without skips**.

The unit-job commands exercised were:

```bash
python -m pip check
python -m pytest -q
python deploy.py --strategy lookahead --output outputs/decision.json
python benchmark.py --instances instances/eval --output outputs/benchmark
python temporal_deploy.py --strategy myopic --output outputs/temporal_decision.json
python scripts/benchmark_temporal.py --max-cases 1 --horizons 1 4 --max-candidates 5 --output outputs/temporal_smoke
```

The RL-job commands exercised were:

```bash
python -m pip check
python -m pytest -q tests/test_environment.py tests/test_temporal_env.py
python train.py --timesteps 512 --output outputs/ppo_smoke
python eval.py --checkpoint outputs/ppo_smoke --output outputs/smoke_decision.json
python train_temporal.py --timesteps 512 --n-steps 64 --output outputs/temporal_ppo_smoke
python scripts/benchmark_temporal.py --max-cases 1 --horizons 1 --checkpoints outputs/temporal_ppo_smoke --output outputs/temporal_rl_smoke
```

Local outputs are retained under `outputs/ci-verification/`, with a directory for
each isolated job. Static lookahead serves 28/28 Mbps on the held-out case.
The static PPO smoke model serves 4/28 Mbps after 512 steps; the temporal PPO
smoke model completes its held-out rollout. These short runs verify the training,
checkpoint loading and decision pipeline, not learning quality or superiority.

## Additional checks and coverage limits

- `actionlint` validates the workflow syntax, matrix expressions and action inputs.
- The sdist and wheel build successfully with `python -m build --no-isolation`.
- The legacy notebook executes with PPO disabled and enabled using a compatible
  legacy checkpoint; all five plots render without cell errors.
- Documentation links and SHA-256 provenance for the legacy figures are checked.
- Gymnasium emits two warnings about unbounded observation-space limits;
  they do not fail its compliance check. Setuptools emits a nonfatal warning
  about the existing license-table metadata format.

Docker is not installed on this machine, so the image build and container runtime
are **not verified**. The current workflow has no Docker, notebook-execution or
distribution-build job; those last two checks were performed locally only.
No hosted Ubuntu run of the corrected working tree has been launched, and no
changes have been committed or pushed by this verification task.
