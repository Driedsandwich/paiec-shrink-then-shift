#!/usr/bin/env python3
"""Regenerate the technical report figures and numeric tables from stored results.

Reads only files under experiments/ in this repository (stored per-seed outputs),
recomputes means, standard deviations and paired differences, and checks them
against the stored summaries. No network access and no experiment execution.
Usage: python tools/make_figures.py --out <dir>
Writes <dir>/figures/{decomposition,brier_curves}.pdf and
<dir>/tables/{main_results,budget_results,same_item_coverage,feature_weights,benchmarks}.tex.
"""
import argparse
import json
import math
import os
import statistics
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "experiments"
E3_INPUTS = EXPERIMENTS / "e3_item"
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-cache-"))


def read(name):
    return json.loads((EXPERIMENTS / name).read_text())


def close(a, b, label):
    if not math.isclose(a, b, rel_tol=1e-11, abs_tol=2e-14):
        raise ValueError(f"Inconsistent input: {label}: {a} != {b}")


def tex(s):
    return s.replace("_", r"\_")


def paired(a, b):
    differences = [x - y for x, y in zip(a, b, strict=True)]
    return statistics.mean(differences), statistics.stdev(differences) / math.sqrt(len(a))


def main(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    d = read("decomposition/holdout_50seeds.json")
    v3 = read("verify_flight3_holdout.json")
    v2 = read("verify_e2_holdout.json")
    ve3 = read("verify_e3_holdout.json")
    e3_records = {strategy: {name: json.loads((E3_INPUTS / f"holdout_{strategy}_{name.lower()}.json").read_text())
                            for name in names}
                  for strategy, names in [("balanced", ("E3", "E2", "E1b")), ("packed", ("E3", "E2"))]}
    e3_results = json.loads((E3_INPUTS / "results.json").read_text())
    verification = json.loads((E3_INPUTS / "verification.json").read_text())
    tuning = read("e1_tuning/results.json")["assignments"]
    features = read("e2_item/weights.json")
    budgets, weights = d["budgets"], d["weights"]
    methods = d["methods"]
    assert budgets == [0, 1, 3, 7, 15, 31]
    assert d["runs"] == 50 and d["seed"] == 1000
    assert d["split"] == "holdout" and d["pairs_strategy"] == "balanced"
    components = ["level_shift", "shape_mismatch", "empirical_fluctuation"]

    # Adapt archived E3 run summaries to the existing table/figure schema.
    # Also check paired seed identities and the full-text baseline replays.
    for strategy, records in e3_records.items():
        for name, record in records.items():
            conditions, runs = record["conditions"], record["runs"]
            assert conditions["seed"] == 1000 and conditions["runs"] == len(runs) == 50
            assert conditions["split"] == "holdout" and conditions["pairs_strategy"] == strategy
            assert conditions["holdout_fraction"] == 0.3
            assert conditions["budgets"] == budgets and conditions["alc_weights"] == weights
            converted = []
            for seed, run in enumerate(runs, 1000):
                assert run["seed"] == seed
                row = {"alc": run["alc"]}
                for n in budgets:
                    row[f"brier_{n}"] = run["brier_by_budget"][str(n)]
                    for c in components:
                        row[f"{c}_{n}"] = run["decomposition_by_budget"][str(n)][c]
                    close(sum(row[f"{c}_{n}"] for c in components), row[f"brier_{n}"], f"{strategy}.{name}.{seed}.{n}")
                close(sum(w * row[f"brier_{n}"] for n, w in zip(budgets, weights)), row["alc"], f"{strategy}.{name}.{seed}.ALC")
                for c in components:
                    close(sum(w * row[f"{c}_{n}"] for n, w in zip(budgets, weights)), run["decomposition_alc"][c], f"{name}.{seed}.{c}.ALC")
                converted.append(row)
            mean = {key: statistics.mean(row[key] for row in converted) for key in converted[0]}
            std = {key: statistics.pstdev(row[key] for row in converted) for key in converted[0]}
            for key in ("alc_mean", "alc_std"):
                close((mean if key == "alc_mean" else std)["alc"], record["summary"][key], f"{strategy}.{name}.{key}")
            for n in budgets:
                close(mean[f"brier_{n}"], record["summary"]["brier_by_budget"][str(n)]["mean"], f"{name}.{n}.mean")
                close(std[f"brier_{n}"], record["summary"]["brier_by_budget"][str(n)]["std"], f"{name}.{n}.std")
                for c in components:
                    close(mean[f"{c}_{n}"], record["decomposition_by_budget"][str(n)][c], f"{name}.{n}.{c}.mean")
            for c in components:
                close(statistics.mean(run["decomposition_alc"][c] for run in runs), record["decomposition_alc"][c], f"{name}.{c}.ALC.mean")
            if name == "E3":
                for n in budgets:
                    coverage = record["same_item_coverage"][str(n)]
                    assert 0 <= coverage["covered"] <= coverage["predictions"]
                    assert coverage["predictions"] == sum(run["evaluation_prediction_calls_by_budget"][str(n)] for run in runs)
                    close(coverage["fraction"], coverage["covered"] / coverage["predictions"], f"{strategy}.{n}.coverage")
                assert record["prediction_timing"]["calls"] == sum(c["predictions"] for c in record["same_item_coverage"].values())
            if strategy == "balanced":
                if name == "E3":
                    methods[name] = {"per_seed": converted, "mean": mean, "std": std}
                else:
                    for row, old in zip(converted, methods[name]["per_seed"], strict=True):
                        for key in row:
                            close(row[key], old[key], f"replayed {name}.{key}")
        comparisons = ("E2", "E1b") if strategy == "balanced" else ("E2",)
        for ref in comparisons:
            a, b = ([run["alc"] for run in records[name]["runs"]] for name in ("E3", ref))
            dm, ds = paired(a, b)
            summary = e3_results["holdout" if strategy == "balanced" else "packed"][f"paired_e3_minus_{ref.lower()}"]
            close(dm, summary["mean"], f"{strategy}.E3-{ref}.mean")
            close(ds, summary["standard_error"], f"{strategy}.E3-{ref}.SE")
            assert sum(x < y for x, y in zip(a, b)) == summary["better_seeds"]
            assert sum(x == y for x, y in zip(a, b)) == summary["equal_seeds"]

    # Recompute every summary used by this draft, not merely the ALC column.
    for method, result in methods.items():
        rows = result["per_seed"]
        assert len(rows) == d["runs"]
        for key in result["mean"]:
            close(statistics.mean(row[key] for row in rows), result["mean"][key], f"{method}.mean.{key}")
            close(statistics.pstdev(row[key] for row in rows), result["std"][key], f"{method}.std.{key}")
        for seed, row in enumerate(rows, d["seed"]):
            close(sum(w * row[f"brier_{n}"] for n, w in zip(budgets, weights)), row["alc"], f"{method}.{seed}.alc")
            for n in budgets:
                close(sum(row[f"{c}_{n}"] for c in components), row[f"brier_{n}"], f"{method}.{seed}.decomposition.{n}")
    for method, other in [("constant_0.5", "constant"), ("empirical_mean", "empirical"), ("E1b", "V2_best")]:
        for a, b in zip(methods[method]["per_seed"], v3["holdout"][other]["alcs"], strict=True):
            close(a["alc"], b, f"cross-file {method}")
    for row, other in zip(methods["E2"]["per_seed"], v2["e2_alcs"], strict=True):
        close(row["alc"], other, "cross-file E2")
    values = {name: [row["alc"] for row in res["per_seed"]] for name, res in methods.items()}
    delta, se = paired(values["E2"], values["E1b"])
    close(delta, v2["paired_diff_mean"], "E2 paired mean")
    close(se, v2["paired_diff_se"], "E2 paired SE")
    assert sum(a < b for a, b in zip(values["E2"], values["E1b"])) == v2["e2_better_seeds"] == 45
    assert ve3["seeds"] == 50
    for a, b in zip(values["E3"], ve3["e3_alcs"], strict=True):
        close(a, b, "cross-file E3 shipping-model verification")
    close(statistics.mean(values["E3"]), ve3["e3_mean"], "E3 verification mean")
    close(statistics.pstdev(values["E3"]), ve3["e3_std"], "E3 verification SD")
    close(statistics.mean(values["E2"]), ve3["e2_mean"], "E3 verification E2 mean")
    delta3, se3 = paired(values["E3"], values["E2"])
    close(delta3, ve3["paired_diff_mean"], "E3 verification paired mean")
    close(se3, ve3["paired_diff_se"], "E3 verification paired SE")
    assert sum(a < b for a, b in zip(values["E3"], values["E2"])) == ve3["e3_better_seeds"] == 40
    assert verification["tests"]["tests_run"] == 84
    assert verification["tests"]["failures"] == verification["tests"]["errors"] == 0

    figures, tables = out / "figures", out / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    x = np.log2(np.asarray(budgets) + 1)
    e1 = methods["E1b"]["mean"]
    fig, ax = plt.subplots(figsize=(6.4, 2.95), layout="constrained")
    bottom = np.zeros(len(budgets))
    for component, label, color in [
        ("empirical_fluctuation", "Empirical fluctuation", "#d8dee8"),
        ("shape_mismatch", "Shape mismatch", "#90b6ce"),
        ("level_shift", "Level mismatch", "#e6b46c"),
    ]:
        y = np.array([e1[f"{component}_{n}"] for n in budgets])
        ax.bar(x, y, width=0.56, bottom=bottom, label=f"E1b: {label}", color=color)
        bottom += y
    ax.plot(x, [methods["empirical_mean"]["mean"][f"brier_{n}"] for n in budgets],
            "o-", color="#a93e40", lw=1.7, ms=4, label="Empirical Mean: Brier")
    ax.plot(x, [e1[f"brier_{n}"] for n in budgets], "s-", color="#183e56", lw=1.7, ms=4, label="E1b: Brier")
    for name, color, marker in [("E2", "#b88124", "x"), ("E3", "#33775b", "^")]:
        ax.plot(x, [methods[name]["mean"][f"brier_{n}"] for n in budgets],
                marker=marker, color=color, lw=1.3, ms=4, label=f"{name}: Brier")
    ax.set(xticks=x, xticklabels=budgets, xlabel=r"Labels per subject–benchmark pair, $n$ ($\log_2(n+1)$ spacing)",
           ylabel="Brier / component", ylim=(0, 0.37))
    ax.legend(fontsize=8, ncols=2, loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    fig.savefig(figures / "decomposition.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 2.8), layout="constrained")
    for name, label, color, marker in [("constant_0.5", "Constant 0.5", "#777777", "."),
                                         ("empirical_mean", "Empirical Mean", "#a93e40", "o"),
                                         ("E1b", "E1b", "#183e56", "s"),
                                         ("E2", "E2", "#b88124", "x"),
                                         ("E3", "E3", "#33775b", "^")]:
        ax.plot(x, [methods[name]["mean"][f"brier_{n}"] for n in budgets],
                marker=marker, color=color, label=label, lw=1.5, ms=5)
    ax.set(xticks=x, xticklabels=budgets, xlabel=r"Labels per pair, $n$ ($\log_2(n+1)$ spacing)", ylabel="Brier", ylim=(0.16, 0.36))
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.18)
    fig.savefig(figures / "brier_curves.pdf")
    plt.close(fig)

    rows = [r"\begin{tabular}{lrlr}", r"\toprule", r"Method & ALC (mean $\pm$ SD) & Reference & $\Delta\pm$ SE \\", r"\midrule"]
    for name, label in [("constant_0.5", "Constant 0.5"), ("empirical_mean", "Empirical Mean"),
                        ("E1", r"E1 ($m=0.5$)"), ("E1b", "E1b"), ("E2", "E2"), ("E3", "E3")]:
        if name == "E1":
            record = v3["holdout"]["V2_dayone_prior05"]
            a, sd, seq = record["alc_mean"], record["alc_std"], record["alcs"]
            close(a, statistics.mean(seq), "E1 mean")
            close(sd, statistics.pstdev(seq), "E1 SD")
        elif name == "E3":
            a, sd, seq = methods[name]["mean"]["alc"], methods[name]["std"]["alc"], values[name]
        else:
            a, sd, seq = methods[name]["mean"]["alc"], methods[name]["std"]["alc"], values[name]
        ref = {"E2": "E1b", "E3": "E2"}.get(name, "empirical_mean")
        ref_label = "Empirical Mean" if ref == "empirical_mean" else ref
        dm, ds = paired(seq, values[ref])
        precision = 6 if name in ("E2", "E3") else 5
        rows.append(f"{label} & ${a:.5f} \\pm {sd:.5f}$ & {ref_label} & ${dm:+.{precision}f} \\pm {ds:.{precision}f}$ " + r"\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (tables / "main_results.tex").write_text("\n".join(rows) + "\n")

    rows = [r"\begin{tabular}{lrrrrrr}", r"\toprule", r"Method / budget $n$ & 0 & 1 & 3 & 7 & 15 & 31 \\", r"\midrule"]
    for name, label in [("constant_0.5", "Constant 0.5"), ("empirical_mean", "Empirical Mean"), ("E1b", "E1b"), ("E2", "E2"), ("E3", "E3")]:
        rows.append(label + " & " + " & ".join(f"{methods[name]['mean'][f'brier_{n}']:.5f}" for n in budgets) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (tables / "budget_results.tex").write_text("\n".join(rows) + "\n")
    coverage = e3_records["balanced"]["E3"]["same_item_coverage"]
    rows = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
            r"Budget $n$ & 0 & 1 & 3 & 7 & 15 & 31 \\", r"\midrule",
            "Covered calls & " + " & ".join(str(coverage[str(n)]["covered"]) for n in budgets) + r" \\",
            r"Coverage (\%) & " + " & ".join(f"{100*coverage[str(n)]['fraction']:.2f}" for n in budgets) + r" \\",
            r"\bottomrule", r"\end{tabular}"]
    (tables / "same_item_coverage.tex").write_text("\n".join(rows) + "\n")
    rows = [r"\begin{tabular}{lrrr}", r"\toprule", r"Feature & Weight $w_j$ & Mean $\mu_j$ & Scale $s_j$ \\", r"\midrule"]
    for name, w, mu, scale in zip(features["feature_names"], features["weights"], features["means"], features["scales"], strict=True):
        rows.append(r"\texttt{" + tex(name) + "} & " + f"{w:+.8f} & {mu:.8f} & {scale:.8f}" + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (tables / "feature_weights.tex").write_text("\n".join(rows) + "\n")
    rows = []
    for key, label in [("tune", "Tuning"), ("holdout", "Holdout")]:
        rows.append(r"\paragraph{" + label + f" ({len(tuning[key])} benchmarks)." + "}")
        rows.append(", ".join(r"\texttt{" + tex(s) + "}" for s in tuning[key]) + ".")
    (tables / "benchmarks.tex").write_text("\n".join(rows) + "\n")
    print("Verified stored statistics, decompositions, weighted ALCs, and cross-file ALCs.")
    print("Generated two PDF figures and five TeX tables.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    main(parser.parse_args().out.resolve())
