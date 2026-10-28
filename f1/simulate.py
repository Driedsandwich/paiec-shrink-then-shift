"""Local, offline approximation of streaming Brier ALC. See README.md."""

import argparse
import copy
from dataclasses import dataclass, field
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import random
import statistics
import sys

# Participant modules can be outside the output folder; never create bytecode there.
sys.dont_write_bytecode = True

import pyarrow.parquet as pq

if __package__:
    from .splits import benchmark_split
else:
    from splits import benchmark_split


BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
SUBJECT_FIELDS = ("normalized_name", "provider", "release_date", "access_date",
                  "harness", "harness_version", "reasoning_effort", "subject_features_extra")
RESPONSE_COLUMNS = ("subject_id", "benchmark_id", "item_id", "trial", "interactors", "response")
MIN_ITEMS = 80
MAX_LABELS = 31


def text_value(value):
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, allow_nan=False)


def anonymous(kind, original):
    return kind + "_" + hashlib.sha256(original.encode()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def seeded_rng(seed, *parts):
    return random.Random(int.from_bytes(hashlib.sha256(canonical([seed, *parts]).encode()).digest(), "big"))


def read_rows(folder, table, columns):
    """Column projection and bounded Arrow batches; never open traces/assets."""
    path = folder / (table + ".parquet")
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=list(columns)):
        yield from batch.to_pylist()


def default_acquire(input, context):
    key = json.dumps([context, input], sort_keys=True, separators=(",", ":"))
    uniform = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64
    return uniform < min(1, context["labels_remaining"] / context["items_remaining"])


class AcquisitionHook:
    """Same signature binding/copy/bool contract as the supplied evaluator.

    Legacy pool ranking is deliberately rejected: F1 implements streaming only.
    A participant TypeError is propagated once, never used to guess a signature.
    """

    def __init__(self, function):
        self.function = function
        self.mode, self.positional, self.keywords = "random", False, ()
        if function is None:
            return
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError) as exc:
            raise ValueError("Cannot inspect acquisition_function signature") from exc
        names = ("prediction", "labeled", "context")
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        keywords = tuple(name for name in names if has_kwargs or (
            name in signature.parameters and signature.parameters[name].kind != inspect.Parameter.POSITIONAL_ONLY))
        shapes = [(False, names), (True, ())]
        if keywords and keywords != names:
            shapes.append((False, keywords))
        for positional, supplied in shapes:
            try:
                if positional:
                    signature.bind(None, None, None, None)
                else:
                    signature.bind(None, **{name: None for name in supplied})
            except TypeError:
                continue
            self.mode, self.positional, self.keywords = "stream", positional, supplied
            return
        raise ValueError("F1 requires a streaming bool acquisition_function; legacy ranking is unsupported")

    @property
    def needs_prediction(self):
        return self.positional or "prediction" in self.keywords

    def __call__(self, input, prediction, labeled, context):
        if self.mode == "random":
            return default_acquire(input, context)
        values = {"prediction": prediction, "labeled": labeled, "context": context}
        if self.positional:
            result = self.function(copy.deepcopy(input), prediction, copy.deepcopy(labeled), copy.deepcopy(context))
        else:
            result = self.function(copy.deepcopy(input), **{k: copy.deepcopy(values[k]) for k in self.keywords})
        if type(result) is not bool:
            raise ValueError("Streaming acquisition_function must return a bool")
        return result


def probability(predict, input, labeled):
    value = float(predict(copy.deepcopy(input), copy.deepcopy(labeled)))
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("predict must return a finite probability in [0, 1]")
    return value


def load_function(path, name):
    path = Path(path).resolve()
    module_name = "f1_participant_" + name + "_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    # Permit ordinary sibling imports, without changing the process working folder.
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    function = getattr(module, name, None)
    if not callable(function):
        raise ValueError(f"{path} must define {name}")
    return function


@dataclass(frozen=True)
class Pair:
    folder: Path
    subject: str
    benchmark: str
    n_items: int

    @property
    def key(self):
        return (self.benchmark, self.subject)

    @property
    def token(self):
        return anonymous("pair", canonical(self.key))


def subjects_in(folder, release_years):
    subjects, excluded = {}, 0
    for row in read_rows(folder, "subjects", ("subject_id", *SUBJECT_FIELDS)):
        date = text_value(row["release_date"])
        if release_years is not None and date[:4] not in release_years:
            excluded += 1
            continue
        subjects[row["subject_id"]] = {key: text_value(row[key]) for key in SUBJECT_FIELDS}
    return subjects, excluded


def response_label(row):
    value = row["response"]
    if value is None:
        return None
    if value not in (0, 1):
        raise ValueError("A selected binary benchmark has a non-binary or non-finite response")
    return int(value)


def item_sets(folder, benchmarks, subjects):
    """Retain distinct IDs, not full response rows, for eligibility/sampling."""
    valid_items = {row["item_id"]: row["benchmark_id"] for row in
                   read_rows(folder, "items", ("item_id", "benchmark_id"))
                   if row["benchmark_id"] in benchmarks}
    groups, nulls = {}, 0
    for row in read_rows(folder, "responses", ("subject_id", "benchmark_id", "item_id", "response")):
        if row["benchmark_id"] not in benchmarks or row["subject_id"] not in subjects:
            continue
        if valid_items.get(row["item_id"]) != row["benchmark_id"]:
            raise ValueError("Response references a missing item or a different benchmark")
        if response_label(row) is None:
            nulls += 1
            continue
        key = (row["benchmark_id"], row["subject_id"])
        groups.setdefault(key, set()).add(row["item_id"])
    return groups, nulls


def inventory(data, benchmarks, split, holdout_fraction, release_years):
    root = Path(data)
    folders = list(root.glob("*/formatted_tables"))
    if (root / "formatted_tables").is_dir():
        folders.append(root / "formatted_tables")
    if not folders:
        raise ValueError("No {dataset}/formatted_tables directories found")
    # Metadata only: process smaller response files first, one dataset at a time.
    folders.sort(key=lambda p: ((p / "responses.parquet").stat().st_size
                              if (p / "responses.parquet").exists() else 0, str(p)))
    pairs, reports, owners = [], [], {}
    found_benchmarks = set()
    for folder in folders:
        report = {"dataset": str(folder.parent.relative_to(root)), "benchmarks": []}
        chosen = set()
        for row in read_rows(folder, "benchmarks", ("benchmark_id", "response_type", "granularity")):
            bid = row["benchmark_id"]
            found_benchmarks.add(bid)
            assigned = benchmark_split(bid, holdout_fraction)
            reason = None
            if benchmarks is not None and bid not in benchmarks:
                reason = "benchmark_filter"
            elif row["response_type"] != "binary" or row["granularity"] != "item":
                reason = "requires_binary_item"
            elif split != "all" and split != assigned:
                reason = "split_filter"
            report["benchmarks"].append({"original_benchmark_id": bid,
                                          "benchmark_id": anonymous("benchmark", bid),
                                          "split": assigned, "excluded_reason": reason})
            if reason is None:
                if bid in owners:
                    raise ValueError(f"Duplicate benchmark_id across datasets: {bid}")
                owners[bid] = folder
                chosen.add(bid)
        if chosen:
            subjects, excluded = subjects_in(folder, release_years)
            groups, nulls = item_sets(folder, chosen, subjects)
            eligible = [Pair(folder, sid, bid, len(items)) for (bid, sid), items in groups.items()
                        if len(items) >= MIN_ITEMS]
            pairs.extend(eligible)
            report.update(eligible_pairs=len(eligible),
                          pairs_below_80=sum(len(items) < MIN_ITEMS for items in groups.values()),
                          excluded_subjects_by_release_year=excluded, ungraded_rows_excluded=nulls)
            del groups, subjects
        reports.append(report)
    if benchmarks is not None and benchmarks - found_benchmarks:
        raise ValueError("Requested benchmarks not found: " + ", ".join(sorted(benchmarks - found_benchmarks)))
    return sorted(pairs, key=lambda p: p.key), reports


def allocate_pairs(pairs, seed, strategy, max_sample):
    order = sorted(pairs, key=lambda p: p.key)
    seeded_rng(seed, "pair_selection").shuffle(order)
    allocations = {}
    if strategy == "balanced":
        selected = order[:max_sample // MIN_ITEMS]
        allocations = {pair: MIN_ITEMS for pair in selected}
        remaining = max_sample - sum(allocations.values())
        while remaining:
            changed = False
            for pair in selected:
                if allocations[pair] < pair.n_items:
                    allocations[pair] += 1
                    remaining -= 1
                    changed = True
                    if remaining == 0:
                        break
            if not changed:
                break
    elif strategy == "packed":
        remaining = max_sample
        for pair in order:
            if remaining < MIN_ITEMS:
                break
            allocations[pair] = min(remaining, pair.n_items)
            remaining -= allocations[pair]
    else:
        raise ValueError("Unknown pairs_strategy")
    return allocations


@dataclass
class PairState:
    pair: Pair
    acquisition: list  # (item_id, input, one seeded recorded label)
    evaluation: list  # (item_id, input, all labels sharing that input)
    evaluation_ids: list
    n_sampled: int
    cursor: int = 0
    acquired_ids: list = field(default_factory=list)
    checkpoints: dict = field(default_factory=dict)


def prepare_states(allocations, seed, release_years):
    states = []
    for folder in sorted({p.folder for p in allocations}, key=str):
        selected = {p.key: p for p in allocations if p.folder == folder}
        subjects, _ = subjects_in(folder, release_years)
        groups, _ = item_sets(folder, {p.benchmark for p in selected.values()},
                              {p.subject for p in selected.values()})
        partitions, observations = {}, {}
        for key, pair in selected.items():
            ids = sorted(groups[key])
            seeded_rng(seed, "items", pair.key).shuffle(ids)
            ids = ids[:allocations[pair]]
            # Sampling and splitting use independent seeded streams.
            seeded_rng(seed, "split", pair.key).shuffle(ids)
            cut = len(ids) // 2
            acquisition, evaluation = ids[:cut], ids[cut:]
            seeded_rng(seed, "candidate_order", pair.key).shuffle(acquisition)
            partitions[key] = (acquisition, evaluation)
            observations[key] = {iid: [] for iid in ids}
        del groups
        wanted_items = {iid for group in observations.values() for iid in group}
        items = {row["item_id"]: row for row in
                 read_rows(folder, "items", ("item_id", "benchmark_id", "content", "item_features"))
                 if row["item_id"] in wanted_items}
        for row in read_rows(folder, "responses", RESPONSE_COLUMNS):
            key = (row["benchmark_id"], row["subject_id"])
            if key not in observations or row["item_id"] not in observations[key]:
                continue
            label = response_label(row)
            if label is not None:
                observations[key][row["item_id"]].append(
                    (row["trial"], text_value(row["interactors"]), label))
        for key, pair in selected.items():
            acq_ids, eval_ids = partitions[key]

            def make_input(iid, interactors):
                item = items[iid]
                return [subjects[pair.subject], {
                    "item_content": text_value(item["content"]),
                    "item_features": text_value(item["item_features"]),
                    "interactors": interactors,
                    "benchmark_id": anonymous("benchmark", pair.benchmark)}]

            acquisition, evaluation = [], []
            for iid in acq_ids:
                rows = sorted(observations[key][iid])
                _, interactors, label = seeded_rng(seed, "trial", pair.key, iid).choice(rows)
                acquisition.append((iid, make_input(iid, interactors), label))
            for iid in eval_ids:
                by_interactors = {}
                for _, interactors, label in sorted(observations[key][iid]):
                    by_interactors.setdefault(interactors, []).append(label)
                for interactors, labels in by_interactors.items():
                    evaluation.append((iid, make_input(iid, interactors), labels))
            states.append(PairState(pair, acquisition, evaluation, sorted(eval_ids), allocations[pair]))
    return sorted(states, key=lambda state: state.pair.key)


def score_pair(evaluation, predictions):
    """Response-weighted empirical identity; never replace row scoring by means."""
    cells = [(predictions[canonical(input)], labels) for _, input, labels in evaluation]
    total = sum(len(labels) for _, labels in cells)
    pbar = math.fsum(p * len(labels) for p, labels in cells) / total
    ybar = math.fsum(sum(labels) for _, labels in cells) / total
    level = (pbar - ybar)**2
    shape = math.fsum(len(ys) * ((p - pbar) - (sum(ys) / len(ys) - ybar))**2
                      for p, ys in cells) / total
    fluctuation = math.fsum(len(ys) * (sum(ys) / len(ys)) * (1 - sum(ys) / len(ys))
                            for _, ys in cells) / total
    brier = math.fsum((p - y)**2 for p, ys in cells for y in ys) / total
    decomposition = {"level_shift": level, "shape_mismatch": shape,
                     "empirical_fluctuation": fluctuation}
    residual = brier - math.fsum(decomposition.values())
    if not math.isclose(residual, 0, abs_tol=1e-12):
        raise ArithmeticError("Brier decomposition identity failed")
    return {"brier": brier, "decomposition": decomposition, "decomposition_residual": residual,
            "scored_response_rows": total, "decomposition_cells": len(cells)}


def simulate_run(states, predict, hook, seed):
    labeled = []
    calls, acquisition_prediction_calls, acquisition_calls = {}, 0, 0
    for budget in BUDGETS:
        # A single trajectory: pause each pair at this checkpoint, then score all.
        for state in states:
            while len(state.acquired_ids) < budget and state.cursor < len(state.acquisition):
                iid, input, label = state.acquisition[state.cursor]
                acquired = len(state.acquired_ids)
                context = {"subject_id": anonymous("subject", state.pair.subject),
                           "benchmark_id": anonymous("benchmark", state.pair.benchmark),
                           "labels_acquired": acquired, "labels_remaining": MAX_LABELS - acquired,
                           "max_labels": MAX_LABELS,
                           "items_remaining": len(state.acquisition) - state.cursor}
                prediction = probability(predict, input, labeled) if hook.needs_prediction else None
                acquisition_prediction_calls += int(hook.needs_prediction)
                query = hook(input, prediction, labeled, context)
                acquisition_calls += 1
                state.cursor += 1
                if query:
                    state.acquired_ids.append(iid)
                    labeled.append([copy.deepcopy(input), label])
        # Cache ONLY inside one checkpoint: evidence changes at the next budget.
        predictions = {}
        for state in states:
            for _, input, _ in state.evaluation:
                key = canonical(input)
                if key not in predictions:
                    predictions[key] = probability(predict, input, labeled)
        calls[str(budget)] = len(predictions)
        for state in states:
            metrics = score_pair(state.evaluation, predictions)
            metrics.update(labels_acquired=len(state.acquired_ids),
                           acquired_item_ids=list(state.acquired_ids), global_labels_count=len(labeled),
                           candidate_cursor=state.cursor,
                           acquisition_exhausted=state.cursor == len(state.acquisition),
                           budget_reached=len(state.acquired_ids) == budget)
            state.checkpoints[str(budget)] = metrics
    results = []
    for state in states:
        results.append({"pair_id": state.pair.token,
                        "subject_id": anonymous("subject", state.pair.subject),
                        "benchmark_id": anonymous("benchmark", state.pair.benchmark),
                        "source_unique_items": state.pair.n_items, "sampled_unique_items": state.n_sampled,
                        "acquisition_item_ids": [iid for iid, _, _ in state.acquisition],
                        "evaluation_item_ids": state.evaluation_ids,
                        "alc": math.fsum(w * state.checkpoints[str(b)]["brier"]
                                         for w, b in zip(WEIGHTS, BUDGETS)),
                        "checkpoints": state.checkpoints})
    return {"seed": seed, "sampled_unique_subject_items": sum(s.n_sampled for s in states),
            "pairs": results, "alc": statistics.mean(p["alc"] for p in results),
            "brier_by_budget": {str(b): statistics.mean(p["checkpoints"][str(b)]["brier"] for p in results)
                                for b in BUDGETS},
            "evaluation_prediction_calls_by_budget": calls,
            "acquisition_prediction_calls": acquisition_prediction_calls,
            "acquisition_calls": acquisition_calls}


def run_simulation(data, predict, acquisition_function=None, *, benchmarks=None, runs=1, seed=0,
                   pairs_strategy="balanced", max_sample=1000, split="all", holdout_fraction=0.2,
                   release_years=("2025", "2026")):
    if runs < 1 or not MIN_ITEMS <= max_sample <= 1000:
        raise ValueError("runs must be positive and max_sample must be in [80, 1000]")
    if split not in ("all", "tune", "holdout"):
        raise ValueError("split must be all, tune or holdout")
    benchmark_split("", holdout_fraction)  # Validate even for an empty input directory.
    release_years = None if release_years is None else set(release_years)
    benchmarks = None if benchmarks is None else set(benchmarks)
    pairs, report = inventory(data, benchmarks, split, holdout_fraction, release_years)
    if not pairs:
        raise ValueError("No eligible subject-benchmark pairs (at least 80 graded distinct items required)")
    hook = AcquisitionHook(acquisition_function)
    results = []
    for index in range(runs):
        run_seed = seed + index
        allocations = allocate_pairs(pairs, run_seed, pairs_strategy, max_sample)
        states = prepare_states(allocations, run_seed, release_years)
        results.append(simulate_run(states, predict, hook, run_seed))
    return {"format": "f1_streaming_alc_v1", "official_equivalence": False,
            "conditions": {"data": str(Path(data).resolve()),
                           "seed": seed, "runs": runs, "run_seed_rule": "seed + run_index (0-based)",
                           "pairs_strategy": pairs_strategy, "max_unique_subject_items": max_sample,
                           "pair_selection": "seeded shuffle of eligible pairs before allocation",
                           "allocation_rule": ("select up to floor(cap/80) pairs; start at 80, then round-robin to capacity"
                                               if pairs_strategy == "balanced" else
                                               "fill each shuffled pair to capacity; stop if fewer than 80 slots remain"),
                           "minimum_distinct_graded_items": MIN_ITEMS,
                           "eligibility_before_cap": True, "selected_pair_minimum_items": MIN_ITEMS,
                           "acquisition_mode": hook.mode, "max_labels_per_pair": MAX_LABELS,
                           "budgets": list(BUDGETS), "alc_weights": list(WEIGHTS),
                           "split": split, "holdout_fraction": holdout_fraction,
                           "split_rule": "sha256(original benchmark_id) / 2**256 < holdout_fraction",
                           "release_years": sorted(release_years) if release_years is not None else "all",
                           "benchmarks": sorted(benchmarks) if benchmarks is not None else "all",
                           "pair_order": "lexicographic (original benchmark_id, original subject_id)",
                           "synchronization": "advance every pair to checkpoint before any evaluation",
                           "odd_split": "floor(n/2) acquisition, ceil(n/2) evaluation",
                           "trial_selection": "one seeded uniform recorded row per acquisition item, fixed before hooks",
                           "ungraded_rows": "exclude; never treat missing outcomes as zero",
                           "exhaustion": "no forced reveals; report actual count and repeat evidence if exhausted",
                           "decomposition_cells": "item_id plus input (split by interactors if different)",
                           "prediction_cache": "distinct full input across all pairs, reset each checkpoint",
                           "runtime": "trusted in-process Python, no worker reset or timeout isolation",
                           "std_definition": "population standard deviation across run-level pair means (ddof=0)"},
            "inventory": report, "eligible_pair_count": len(pairs), "runs": results,
            "summary": {"alc_mean": statistics.mean(r["alc"] for r in results),
                        "alc_std": statistics.pstdev(r["alc"] for r in results),
                        "brier_by_budget": {str(b): {
                            "mean": statistics.mean(r["brier_by_budget"][str(b)] for r in results),
                            "std": statistics.pstdev(r["brier_by_budget"][str(b)] for r in results)}
                            for b in BUDGETS}}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--labeling", type=Path)
    parser.add_argument("--benchmarks", help="comma-separated original benchmark IDs")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pairs-strategy", choices=("balanced", "packed"), default="balanced")
    parser.add_argument("--max-sample", type=int, default=1000)
    parser.add_argument("--split", choices=("tune", "holdout", "all"), default="all")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--release-years", default="2025,2026", help="comma-separated years, or all")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = run_simulation(
            args.data, load_function(args.model, "predict"),
            load_function(args.labeling, "acquisition_function") if args.labeling else None,
            benchmarks=set(args.benchmarks.split(",")) if args.benchmarks else None,
            runs=args.runs, seed=args.seed, pairs_strategy=args.pairs_strategy, max_sample=args.max_sample,
            split=args.split, holdout_fraction=args.holdout_fraction,
            release_years=None if args.release_years == "all" else args.release_years.split(","))
        result["conditions"].update(model=str(args.model), labeling=str(args.labeling) if args.labeling else None)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    except (ValueError, OSError, TypeError) as exc:
        parser.exit(2, f"F1 failed: {exc}\n")
    print(f"F1: {args.runs} runs, ALC mean={result['summary']['alc_mean']:.8f}, "
          f"std={result['summary']['alc_std']:.8f}; {args.out}")


if __name__ == "__main__":
    main()
