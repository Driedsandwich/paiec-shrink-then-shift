"""Per-budget Brier and its decomposition on holdout benchmarks (full F1 path).

Brier = level_shift + shape_mismatch + empirical_fluctuation, computed per pair
and budget by f1.simulate.score_pair, then averaged over pairs and seeds.
"""

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

from f1.simulate import BUDGETS, WEIGHTS, run_simulation
import variants

COMPONENTS = ("level_shift", "shape_mismatch", "empirical_fluctuation")


def load_predict(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.predict


def summarize(result):
    per_seed = []
    for run in result["runs"]:
        row = {"alc": run["alc"]}
        for budget in BUDGETS:
            pairs = [p["checkpoints"][str(budget)] for p in run["pairs"]]
            row[f"brier_{budget}"] = statistics.mean(p["brier"] for p in pairs)
            for name in COMPONENTS:
                row[f"{name}_{budget}"] = statistics.mean(p["decomposition"][name] for p in pairs)
        per_seed.append(row)
    keys = per_seed[0].keys()
    return {"mean": {k: statistics.mean(r[k] for r in per_seed) for k in keys},
            "std": {k: statistics.pstdev(r[k] for r in per_seed) for k in keys},
            "per_seed": per_seed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    methods = {
        "constant_0.5": variants.constant,
        "empirical_mean": variants.empirical,
        "E1b": load_predict(root / "submission_e1b/model.py"),
        "E2": load_predict(root / "submission_e2/model.py"),
    }
    out = {"split": "holdout", "holdout_fraction": 0.3, "runs": args.runs, "seed": args.seed,
           "pairs_strategy": "balanced", "budgets": list(BUDGETS), "weights": list(WEIGHTS),
           "methods": {}}
    for name, predict in methods.items():
        start = time.time()
        result = run_simulation(args.data, predict, split="holdout", holdout_fraction=0.3,
                                runs=args.runs, seed=args.seed, pairs_strategy="balanced")
        out["methods"][name] = summarize(result)
        out["methods"][name]["seconds"] = round(time.time() - start, 1)
        print(name, round(out["methods"][name]["mean"]["alc"], 6), out["methods"][name]["seconds"],
              flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
