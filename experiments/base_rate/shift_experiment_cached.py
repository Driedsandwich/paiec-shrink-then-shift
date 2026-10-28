"""Cached version of shift_experiment.py for label-only predictors (E1, E1b, E4).

Observations of the population's pairs are read once. Per seed, pair selection,
item sampling, the acquisition/evaluation split, candidate order and the
recorded-trial choice use the same seeded streams as f1.simulate.prepare_states.
Item content and features are also read once, so every input is identical to
the full path (the default random acquisition hashes the full input, including
the item text). Equivalence with the full path is checked by --verify, which
compares per-seed ALC of E1 and E4 (expected difference 0).
"""
import argparse, copy, importlib.util, json, statistics, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import f1.simulate as sim


def load(path):
    spec = importlib.util.spec_from_file_location("m_" + Path(path).parent.name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def preload(pairs, years):
    obs = {}  # pair.key -> {item_id: [(trial, interactors, label)]}
    subjects_by_folder = {}
    for folder in sorted({p.folder for p in pairs}, key=str):
        selected = {p.key for p in pairs if p.folder == folder}
        subjects_by_folder[folder], _ = sim.subjects_in(folder, years)
        for row in sim.read_rows(folder, "responses", sim.RESPONSE_COLUMNS):
            key = (row["benchmark_id"], row["subject_id"])
            if key not in selected:
                continue
            label = sim.response_label(row)
            if label is not None:
                obs.setdefault(key, {}).setdefault(row["item_id"], []).append(
                    (row["trial"], sim.text_value(row["interactors"]), label))
    content = {}
    for folder in sorted({p.folder for p in pairs}, key=str):
        wanted = {iid for p in pairs if p.folder == folder for iid in obs.get(p.key, {})}
        for row in sim.read_rows(folder, "items", ("item_id", "benchmark_id", "content", "item_features")):
            if row["item_id"] in wanted:
                content[row["item_id"]] = (sim.text_value(row["content"]), sim.text_value(row["item_features"]))
    return obs, subjects_by_folder, content


def states_from_cache(allocations, seed, obs, subjects_by_folder, content):
    states = []
    for pair in sorted(allocations, key=lambda p: p.key):
        key = pair.key
        ids = sorted(obs[key])
        sim.seeded_rng(seed, "items", key).shuffle(ids)
        ids = ids[:allocations[pair]]
        sim.seeded_rng(seed, "split", key).shuffle(ids)
        cut = len(ids) // 2
        acq_ids, eval_ids = ids[:cut], ids[cut:]
        sim.seeded_rng(seed, "candidate_order", key).shuffle(acq_ids)
        subject = subjects_by_folder[pair.folder][pair.subject]

        def make_input(iid, interactors):
            return [subject, {"item_content": content[iid][0], "item_features": content[iid][1],
                              "interactors": interactors,
                              "benchmark_id": sim.anonymous("benchmark", pair.benchmark)}]

        acquisition, evaluation = [], []
        for iid in acq_ids:
            rows = sorted(obs[key][iid])
            _, interactors, label = sim.seeded_rng(seed, "trial", key, iid).choice(rows)
            acquisition.append((iid, make_input(iid, interactors), label))
        for iid in eval_ids:
            by_interactors = {}
            for _, interactors, label in sorted(obs[key][iid]):
                by_interactors.setdefault(interactors, []).append(label)
            for interactors, labels in by_interactors.items():
                evaluation.append((iid, make_input(iid, interactors), labels))
        states.append(sim.PairState(pair, acquisition, evaluation, sorted(eval_ids), allocations[pair]))
    return states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--rates", required=True)
    ap.add_argument("--split", choices=("tune", "holdout"), required=True)
    ap.add_argument("--condition", choices=("standard", "low"), required=True)
    ap.add_argument("--strengths", default="1,2,4,8,16,32")
    ap.add_argument("--runs", type=int, default=50); ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--verify", type=int, default=0, help="also run the full path for this many seeds")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    years = {"2025", "2026"} if a.condition == "standard" else None
    rates = json.loads(Path(a.rates).read_text())[f"{a.split}_{'2025-26' if years else 'all'}"]["rates"]
    pairs, _ = sim.inventory(a.data, None, a.split, 0.3, years)
    if a.condition == "low":
        pairs = [p for p in pairs if rates[f"{p.benchmark}|{p.subject}"] < 0.55]
    e1, e1b, e4 = (load(ROOT / f"submission_{n}/model.py") for n in ("e1", "e1b", "e4"))
    methods = {"E1": e1.predict, "E1b": e1b.predict}
    for c in [float(x) for x in a.strengths.split(",") if x]:
        methods[f"E4_c{c:g}"] = (lambda c: (lambda inp, lab=None: e4.predict_with_strength(inp, lab, c)))(c)
    hook = sim.AcquisitionHook(None)
    t0 = time.time()
    obs, subjects_by_folder, content = preload(pairs, years)
    t_load = time.time() - t0
    per_seed = {m: [] for m in methods}
    rates_seed = []
    for i in range(a.runs):
        s = a.seed + i
        states = states_from_cache(sim.allocate_pairs(pairs, s, "balanced", 1000), s, obs, subjects_by_folder, content)
        rates_seed.append(statistics.mean(sum(sum(l) for _, _, l in st.evaluation) / sum(len(l) for _, _, l in st.evaluation)
                                          for st in states))
        for m, f in methods.items():
            r = sim.simulate_run(copy.deepcopy(states), f, hook, s)
            per_seed[m].append({"alc": r["alc"], **{f"b{k}": v for k, v in r["brier_by_budget"].items()}})
    verify = None
    if a.verify:
        verify = []
        for i in range(a.verify):
            s = a.seed + i
            full = sim.prepare_states(sim.allocate_pairs(pairs, s, "balanced", 1000), s, years)
            for m in ("E1", next(k for k in methods if k.startswith("E4"))):
                r = sim.simulate_run(copy.deepcopy(full), methods[m], hook, s)
                verify.append({"seed": s, "method": m, "full": r["alc"], "cached": per_seed[m][i]["alc"],
                               "abs_diff": abs(r["alc"] - per_seed[m][i]["alc"])})
    out = {"split": a.split, "condition": a.condition, "pairs_in_population": len(pairs), "runs": a.runs,
           "seed": a.seed, "load_seconds": round(t_load, 1), "seconds": round(time.time() - t0, 1),
           "population_mean_rate_of_sampled_pairs": statistics.mean(rates_seed), "verify": verify, "methods": {}}
    for m, rows in per_seed.items():
        out["methods"][m] = {"alc_mean": statistics.mean(r["alc"] for r in rows),
                             "alc_std": statistics.pstdev(r["alc"] for r in rows),
                             "by_budget": {k: statistics.mean(r[k] for r in rows) for k in rows[0] if k.startswith("b")},
                             "per_seed_alc": [r["alc"] for r in rows]}
        if m != "E1":
            d = [x - y for x, y in zip(out["methods"][m]["per_seed_alc"], out["methods"]["E1"]["per_seed_alc"])]
            out["methods"][m]["vs_E1"] = {"mean_diff": statistics.mean(d), "se": statistics.pstdev(d) / len(d) ** 0.5,
                                          "wins": sum(x < 0 for x in d), "n": len(d)}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(a.split, a.condition, "pairs", len(pairs), "rate", round(out["population_mean_rate_of_sampled_pairs"], 3),
          "load", out["load_seconds"], "sec", out["seconds"])
    if verify:
        print("  verify max|full-cached|", max(v["abs_diff"] for v in verify), "n", len(verify))
    for m, v in out["methods"].items():
        print(f"  {m:8s} alc {v['alc_mean']:.5f}", ("" if m == "E1" else
              f"d {v['vs_E1']['mean_diff']:+.5f}±{v['vs_E1']['se']:.5f} wins {v['vs_E1']['wins']}/{v['vs_E1']['n']}"))


if __name__ == "__main__":
    main()
