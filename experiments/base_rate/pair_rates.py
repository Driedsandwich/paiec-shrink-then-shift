"""Equal-weight mean success rate of eligible subject-benchmark pairs (public data only).

Writes per-pair rates for the shift experiment. No item content is read.
"""
import json, statistics, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from f1.simulate import inventory, read_rows, response_label

data = sys.argv[1]; out = Path(sys.argv[2])
res = {}
for split in ("tune", "holdout"):
    for years in (("2025", "2026"), None):
        pairs, _ = inventory(data, None, split, 0.3, None if years is None else set(years))
        want = {p.key: p for p in pairs}
        sums = defaultdict(lambda: defaultdict(list))
        for folder in sorted({p.folder for p in pairs}, key=str):
            for row in read_rows(folder, "responses", ("subject_id", "benchmark_id", "item_id", "response")):
                k = (row["benchmark_id"], row["subject_id"])
                if k in want:
                    y = response_label(row)
                    if y is not None:
                        sums[k][row["item_id"]].append(y)
        rates = {f"{b}|{s}": statistics.mean(statistics.mean(v) for v in items.values())
                 for (b, s), items in sums.items()}
        tag = f"{split}_{'2025-26' if years else 'all'}"
        r = list(rates.values())
        res[tag] = {"pairs": len(r), "benchmarks": len({k.split('|')[0] for k in rates}),
                    "mean": statistics.mean(r), "median": statistics.median(r),
                    "share_below_0.55": sum(x < 0.55 for x in r) / len(r), "rates": rates}
        print(tag, res[tag]["pairs"], res[tag]["benchmarks"], round(res[tag]["mean"], 4),
              round(res[tag]["median"], 4), round(res[tag]["share_below_0.55"], 3), flush=True)
out.write_text(json.dumps(res, indent=1))
