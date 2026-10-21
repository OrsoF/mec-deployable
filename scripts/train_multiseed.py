"""Independent PPO trainings: each seed gets a NEW model and checkpoint.

Run both static or temporal experiments from the command line. The manifest
records failures, and *never* presents evaluation seeds as training seeds.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("static", "temporal"), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37, 47, 59])
    ap.add_argument("--budgets", type=int, nargs="+", default=[100000, 500000, 1000000])
    ap.add_argument("--output", type=Path, default=Path("outputs/multiseed"))
    ap.add_argument("--action-mode", choices=("atomic", "hybrid"), default="hybrid")
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args()
    if any(x < 1 for x in args.budgets) or len(set(args.seeds)) != len(args.seeds):
        ap.error("Budgets must be positive and training seeds distinct")
    records = []
    args.output.mkdir(parents=True, exist_ok=True)
    for budget in args.budgets:
        for seed in args.seeds:
            label = f"{args.mode}_{args.action_mode if args.mode == 'static' else 'periodic'}_{budget}_seed{seed}"
            checkpoint = args.output / label
            command = [
                sys.executable,
                str(
                    ROOT
                    / ("train.py" if args.mode == "static" else "train_temporal.py")
                ),
                "--seed",
                str(seed),
                "--timesteps",
                str(budget),
                "--output",
                str(checkpoint),
            ]
            if args.mode == "static":
                command.extend(["--action-mode", args.action_mode])
            began = perf_counter()
            logfile = args.output / f"{label}.log"
            print("RUN", label, flush=True)
            with logfile.open("w", encoding="utf-8") as stream:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            record = {
                "mode": args.mode,
                "seed": seed,
                "budget": budget,
                "action_mode": args.action_mode
                if args.mode == "static"
                else "periodic",
                "checkpoint": str(checkpoint),
                "log": str(logfile),
                "exit_code": result.returncode,
                "wall_seconds": round(perf_counter() - began, 3),
                "completed": result.returncode == 0,
            }
            records.append(record)
            (args.output / "manifest.json").write_text(
                json.dumps(records, indent=2) + "\n", encoding="utf-8"
            )
            if result.returncode and args.fail_fast:
                raise SystemExit(result.returncode)
    if any(not r["completed"] for r in records):
        raise SystemExit(
            "One or more training jobs failed; check manifest.json and logs"
        )


if __name__ == "__main__":
    main()
