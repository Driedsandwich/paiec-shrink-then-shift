"""Write E4_RESULTS.md from tune_*.json and holdout_*.json (numbers only, no IDs)."""
import json, statistics, sys
from pathlib import Path
E = Path(__file__).resolve().parent
out = Path(sys.argv[1])
L = ["# E4 local results (shift experiment)", "",
     "Simulator: f1 streaming ALC, balanced allocation (cap 1,000 unique subject-item pairs), default random acquisition, holdout_fraction 0.3. Cached runner `experiments/base_rate/shift_experiment_cached.py`; per-seed ALC equals the full path on the verified seeds (max |difference| shown per file).",
     "Paired SE = sample SD of same-seed ALC differences (row minus E1) divided by sqrt(n); ALC std = population SD over seeds. --verify 2 checks two seeds per condition (E1 and one E4 setting each).",
     "Conditions: standard = 2025-2026 subjects, all eligible pairs; low = all release years, only pairs whose public success rate is below 0.55. Methods share the same pairs and acquisition trajectory within a seed (paired).", ""]
for split, seeds in (("tune", "0-29"), ("holdout", "1000-1049")):
    for cond in ("standard", "low"):
        f = E / f"{split}_{cond}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        ver = max(v["abs_diff"] for v in d["verify"]) if d.get("verify") else None
        L += [f"## {split} / {cond} (seeds {seeds}, {d['runs']} runs, population {d['pairs_in_population']} pairs, mean success rate of sampled pairs {d['population_mean_rate_of_sampled_pairs']:.3f}, verify max diff {ver})", "",
              "| Method | ALC mean | ALC std | minus E1 (mean ± SE) | seeds better than E1 | B0 | B1 | B3 | B7 | B15 | B31 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for m, v in d["methods"].items():
            vs = v.get("vs_E1")
            b = v["by_budget"]
            if vs:
                if "se_sample" in vs:
                    se_sample = vs["se_sample"]
                else:
                    differences = [x - y for x, y in zip(v["per_seed_alc"], d["methods"]["E1"]["per_seed_alc"])]
                    se_sample = statistics.stdev(differences) / len(differences) ** 0.5 if len(differences) > 1 else 0.0
            L.append(f"| {m} | {v['alc_mean']:.5f} | {v['alc_std']:.5f} | "
                     + (f"{vs['mean_diff']:+.5f} ± {se_sample:.5f} | {vs['wins']}/{vs['n']}" if vs else "— | —")
                     + " | " + " | ".join(f"{b['b'+k]:.4f}" for k in ("0", "1", "3", "7", "15", "31")) + " |")
        L.append("")
out.write_text("\n".join(L) + "\n")
print(out.read_text())
