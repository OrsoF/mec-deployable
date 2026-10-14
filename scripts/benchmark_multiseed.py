"""STATIC evaluation across independent PPO checkpoints and inference budgets.

Unlike benchmark.py (one model with varying inference RNG), this script groups
by the TRAINING seed in checkpoint metadata. Evaluation seed and 96-step episode
budget are recorded separately. A zero-checkpoint run benchmarks baselines.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean,pstdev
from network_ppo.instance_io import load_instance
from network_ppo.inference import DecisionEngine


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instances",type=Path,default=Path("instances/eval"))
    ap.add_argument("--checkpoints",type=Path,nargs="*",default=[])
    ap.add_argument("--output",type=Path,default=Path("outputs/static_multiseed_eval"))
    ap.add_argument("--episode-budgets",type=int,nargs="+",default=[1,4,16])
    ap.add_argument("--eval-seeds",type=int,nargs="+",default=[101,202,303])
    ap.add_argument("--stochastic",action="store_true")
    args=ap.parse_args()
    if any(x<1 for x in args.episode_budgets):
        ap.error("Positive budgets required")
    files=sorted(args.instances.glob('*.json'))
    if not files:
        ap.error('No static evaluation cases')
    cases=[load_instance(x) for x in files]
    rows=[]
    for case,file in zip(cases,files):
        for method in ("greedy","lookahead","multistart"):
            out=DecisionEngine(strategy=method).decide(case)
            rows.append({"instance":case.id,"method":method,"training_seed":"",
                         "evaluation_seed":"","inference_episodes":0,
                         "policy_steps":0,"score_mbps":out["total_served_mbps"],
                         "wall_ms":out["decision_wall_ms"],"feasible":out["feasible"]})
    if args.checkpoints:
        from network_ppo.inference import PolicyRuntime
        for folder in args.checkpoints:
            runtime=PolicyRuntime(folder)
            train_seed=runtime.metadata["seed"]
            for case,file in zip(cases,files):
                checksum=hashlib.sha256(file.read_bytes()).hexdigest()
                if checksum in runtime.metadata.get("training_sha256",{}).values():
                    ap.error(f"Evaluation instance {file.name} overlaps training")
                # Runtime loaded once; call predict repeatedly without checkpoint reload.
                for episodes in args.episode_budgets:
                    for eval_seed in args.eval_seeds if args.stochastic else [args.eval_seeds[0]]:
                        found=runtime.predict(case,episodes=episodes,stochastic=args.stochastic,seed=eval_seed)
                        from network_ppo.constraints import assess
                        from network_ppo.paths import PathCatalog
                        report=assess(case,found.solution,PathCatalog(case,runtime.metadata["paths"]))
                        rows.append({"instance":case.id,"method":("ppo_stochastic" if args.stochastic else "ppo_deterministic"),
                            "training_seed":train_seed,"evaluation_seed":eval_seed,"inference_episodes":episodes,
                            "policy_steps":found.steps,"score_mbps":report.total_mbps,
                            "wall_ms":found.elapsed_ms,"feasible":report.feasible})
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/"results.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    groups={}
    for r in rows:
        key=(r['method'],r['inference_episodes'])
        groups.setdefault(key,[]).append(r)
    summary=[{"method":method,"inference_episodes":budget,"n":len(items),
              "mean_mbps":mean(x['score_mbps'] for x in items),
              "std_mbps":pstdev(x['score_mbps'] for x in items),
              "mean_runtime_ms":mean(x['wall_ms'] for x in items)}
             for (method,budget),items in groups.items()]
    (args.output/"summary.json").write_text(json.dumps({"protocol":{
        "note":"For confidence intervals, resample independent TRAINING seeds, not inference episodes.",
        "ppo_training_seeds":[str(x) for x in args.checkpoints],
        "eval_seeds":args.eval_seeds,"stochastic":args.stochastic,
        "wall_clock":"model preloaded, includes environment setup/inference/feasibility"},
        "results":summary},indent=2)+"\n",encoding="utf-8")
    for r in summary:
        print(f"{r['method']:20s} budget={r['inference_episodes']:2} score="
              f"{r['mean_mbps']:.2f} +/- {r['std_mbps']:.2f} Mbps; {r['mean_runtime_ms']:.1f}ms")

if __name__=='__main__':
    main()
