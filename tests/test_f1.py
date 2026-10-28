"""Offline synthetic contract tests. All temporary writes stay in the workspace."""

import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from f1 import simulate as sim
from f1.splits import benchmark_split


ROOT = Path(__file__).resolve().parents[1]


def write_table(folder, name, rows):
    pq.write_table(pa.Table.from_pylist(rows), folder / (name + ".parquet"))


def build_data(root):
    """Three benchmarks, 101 items each, two eligible and two excluded subjects."""
    for b in range(3):
        bid = f"bench-{b}"
        folder = root / bid / "formatted_tables"
        folder.mkdir(parents=True)
        write_table(folder, "benchmarks", [{"benchmark_id": bid, "response_type": "binary",
                                           "granularity": "item", "description": "unused"}])
        subjects = []
        for sid in ("s0", "s1", "old", "short"):
            row = {field: None for field in sim.SUBJECT_FIELDS}
            row.update(subject_id=sid, normalized_name=sid, provider="example",
                       release_date="2024-01" if sid == "old" else "2025-06")
            subjects.append(row)
        write_table(folder, "subjects", subjects)
        write_table(folder, "items", [{"item_id": f"{bid}-i{i:03}", "benchmark_id": bid,
                                       "content": f"task {i:03}", "item_features": None,
                                       "grading_criterion": "MUST NOT BE READ"} for i in range(101)])
        responses = []
        for sid in ("s0", "s1", "old", "short"):
            for i in range(79 if sid == "short" else 101):
                for trial in range(1, 4 if i % 9 == 0 else 2):
                    responses.append({"subject_id": sid, "benchmark_id": bid,
                                      "item_id": f"{bid}-i{i:03}", "trial": trial,
                                      "response": float((i + trial + int(sid == "s1")) % 2),
                                      "interactors": None, "test_condition": "unused"})
        # An ungraded attempt must not be invented as a failure or counted in scoring.
        responses.append({"subject_id": "s0", "benchmark_id": bid, "item_id": f"{bid}-i000",
                          "trial": 4, "response": None, "interactors": None, "test_condition": None})
        write_table(folder, "responses", responses)
        # These are intentionally invalid parquet: opening them must fail the test.
        (folder / "traces.parquet").write_bytes(b"DO NOT OPEN")
        (folder / "assets.parquet").write_bytes(b"DO NOT OPEN")


class F1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix=".f1-test-", dir=ROOT)
        cls.work = Path(cls.temporary.name)
        cls.data = cls.work / "data"
        build_data(cls.data)
        cls.constant = sim.run_simulation(cls.data, lambda input, labeled: 0.5,
                                         runs=2, seed=51, max_sample=240)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_constant_half_exact_alc_and_pair_mean(self):
        self.assertEqual(self.constant["summary"]["alc_mean"], 0.25)
        self.assertEqual(self.constant["summary"]["alc_std"], 0)
        for run in self.constant["runs"]:
            self.assertEqual(run["alc"], 0.25)
            for pair in run["pairs"]:
                self.assertEqual(pair["alc"], 0.25)
                for checkpoint in pair["checkpoints"].values():
                    self.assertEqual(checkpoint["brier"], 0.25)

    def test_nested_budgets_and_synchronized_global_evidence(self):
        for run in self.constant["runs"]:
            for pair in run["pairs"]:
                previous = []
                for budget in sim.BUDGETS:
                    cp = pair["checkpoints"][str(budget)]
                    ids = cp["acquired_item_ids"]
                    self.assertEqual(ids[:len(previous)], previous)
                    self.assertEqual(len(ids), budget)
                    self.assertEqual(cp["global_labels_count"], len(run["pairs"]) * budget)
                    self.assertTrue(cp["budget_reached"])
                    self.assertEqual(set(ids) & set(pair["evaluation_item_ids"]), set())
                    previous = ids

    def test_all_evaluation_response_rows_are_scored(self):
        # Independently count original response records, including all three trials.
        raw_counts = {}
        for path in self.data.glob("*/formatted_tables/responses.parquet"):
            for row in pq.read_table(path).to_pylist():
                if row["response"] is not None:
                    key = (sim.anonymous("subject", row["subject_id"]), row["item_id"])
                    raw_counts[key] = raw_counts.get(key, 0) + 1
        for run in self.constant["runs"]:
            for pair in run["pairs"]:
                expected = sum(raw_counts[pair["subject_id"], iid] for iid in pair["evaluation_item_ids"])
                self.assertGreater(expected, len(pair["evaluation_item_ids"]))
                for cp in pair["checkpoints"].values():
                    self.assertEqual(cp["scored_response_rows"], expected)

    def test_default_acquisition_matches_exact_hash_formula_in_trajectory(self):
        decisions = []
        original = sim.default_acquire

        def spy(input, context):
            result = original(input, context)
            decisions.append((copy.deepcopy(input), copy.deepcopy(context), result))
            return result

        with patch.object(sim, "default_acquire", spy):
            sim.run_simulation(self.data, lambda x, y: 0.5, max_sample=80, seed=7)
        self.assertGreater(len(decisions), 31)
        for input, context, actual in decisions:
            key = json.dumps([context, input], sort_keys=True, separators=(",", ":"))
            u = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64
            self.assertEqual(actual, u < context["labels_remaining"] / context["items_remaining"])
            self.assertEqual(context["labels_remaining"], 31 - context["labels_acquired"])
            self.assertEqual(set(context), {"subject_id", "benchmark_id", "labels_acquired",
                                            "labels_remaining", "max_labels", "items_remaining"})

    def test_acquisition_has_no_evaluation_items_unrevealed_or_future_labels(self):
        accepted, calls, prediction_inputs = [], [], []

        def predictor(input, labeled):
            self.assertEqual([sim.canonical(entry[0]) for entry in labeled], accepted)
            prediction_inputs.append((copy.deepcopy(input), len(labeled)))
            # Mutation cannot corrupt simulator evidence or another call.
            labeled.clear()
            input[0].clear()
            return 0.4

        def acquisition(input, prediction=None, labeled=None, context=None):
            self.assertEqual(prediction, 0.4)
            self.assertEqual([sim.canonical(entry[0]) for entry in labeled], accepted)
            self.assertTrue(all(entry[1] in (0, 1) for entry in labeled))
            self.assertEqual(len(input), 2)
            self.assertEqual(set(input[0]), set(sim.SUBJECT_FIELDS))
            self.assertEqual(set(input[1]), {"item_content", "item_features", "interactors", "benchmark_id"})
            self.assertTrue(all(isinstance(v, str) for part in input for v in part.values()))
            calls.append((copy.deepcopy(input), copy.deepcopy(context)))
            accepted.append(sim.canonical(input))
            labeled.append([input, "mutation"])
            context.clear()
            input[0].clear()
            return True

        result = sim.run_simulation(self.data, predictor, acquisition, max_sample=240, seed=11)
        pairs = {(p["subject_id"], p["benchmark_id"]): p for p in result["runs"][0]["pairs"]}
        for input, context in calls:
            pair = pairs[context["subject_id"], context["benchmark_id"]]
            # Fixture item IDs end in iNNN; input text ends in the same NNN.
            evaluation_numbers = {iid[-3:] for iid in pair["evaluation_item_ids"]}
            self.assertNotIn(input[1]["item_content"][-3:], evaluation_numbers)
        self.assertEqual(len(accepted), len(pairs) * 31)
        self.assertTrue(prediction_inputs)

    def test_empirical_decomposition_equals_observed_brier(self):
        def predictor(input, labeled):
            return 0.15 + 0.1 * (int(input[1]["item_content"][-3:]) % 7)

        result = sim.run_simulation(self.data, predictor, max_sample=101, seed=3)
        for pair in result["runs"][0]["pairs"]:
            self.assertEqual(len(pair["acquisition_item_ids"]), 50)
            self.assertEqual(len(pair["evaluation_item_ids"]), 51)
            for cp in pair["checkpoints"].values():
                parts = cp["decomposition"]
                self.assertGreater(parts["empirical_fluctuation"], 0)
                self.assertGreater(parts["shape_mismatch"], 0)
                self.assertAlmostEqual(sum(parts.values()), cp["brier"], places=14)
                self.assertLess(abs(cp["decomposition_residual"]), 1e-14)

    def test_formula_on_hand_computable_repeated_trials(self):
        # Item A: p=.2, outcomes 0,1,1; item B: p=.8, outcome 0.
        a, b = [{}, {"q": "A"}], [{}, {"q": "B"}]
        result = sim.score_pair([("A", a, [0, 1, 1]), ("B", b, [0])],
                                {sim.canonical(a): 0.2, sim.canonical(b): 0.8})
        self.assertAlmostEqual(result["brier"], (0.04 + 0.64 + 0.64 + 0.64) / 4)
        self.assertAlmostEqual(result["decomposition"]["level_shift"], (0.35 - 0.5)**2)
        self.assertAlmostEqual(result["decomposition"]["empirical_fluctuation"], 1 / 6)
        self.assertAlmostEqual(sum(result["decomposition"].values()), result["brier"])

    def test_nonconstant_row_scoring_and_equal_pair_aggregation(self):
        def predict(input, labeled):
            return 0.9 if input[0]["normalized_name"] == "s0" else 0.2

        result = sim.run_simulation(self.data, predict, max_sample=240, seed=51)
        run = result["runs"][0]
        expected_pairs, all_errors = [], []
        for pair in run["pairs"]:
            errors = []
            for path in self.data.glob("*/formatted_tables/responses.parquet"):
                for row in pq.read_table(path).to_pylist():
                    if (sim.anonymous("subject", row["subject_id"]) == pair["subject_id"]
                            and row["item_id"] in pair["evaluation_item_ids"] and row["response"] is not None):
                        p = 0.9 if row["subject_id"] == "s0" else 0.2
                        errors.append((p - row["response"])**2)
            expected = math.fsum(errors) / len(errors)
            expected_pairs.append(expected)
            all_errors.extend(errors)
            self.assertAlmostEqual(pair["alc"], expected, places=14)
            for cp in pair["checkpoints"].values():
                self.assertAlmostEqual(cp["brier"], expected, places=14)
        self.assertAlmostEqual(run["alc"], statistics.mean(expected_pairs), places=14)
        self.assertNotAlmostEqual(run["alc"], statistics.mean(all_errors), places=10)

    def test_cap_is_global_across_more_than_twelve_pairs(self):
        pairs = [sim.Pair(Path("dataset"), f"s{i}", "b", 101) for i in range(30)]
        for strategy in ("balanced", "packed"):
            allocated = sim.allocate_pairs(pairs, 7, strategy, 1000)
            self.assertLessEqual(sum(allocated.values()), 1000)
            self.assertLessEqual(len(allocated), 12)
            self.assertTrue(all(80 <= n <= 101 for n in allocated.values()))
        allocated = sim.allocate_pairs(pairs, 7, "balanced", 1000)
        self.assertEqual(sum(allocated.values()), 1000)
        self.assertEqual(len(allocated), 12)

    def test_interactor_variants_share_item_split_and_decompose_per_input(self):
        a, b = [{}, {"interactors": "A"}], [{}, {"interactors": "B"}]
        result = sim.score_pair([("same_item", a, [0, 1]), ("same_item", b, [1, 1, 0])],
                                {sim.canonical(a): 0.2, sim.canonical(b): 0.9})
        self.assertEqual(result["scored_response_rows"], 5)
        self.assertEqual(result["decomposition_cells"], 2)
        expected = (0.2**2 + 0.8**2 + 0.1**2 + 0.1**2 + 0.9**2) / 5
        self.assertAlmostEqual(result["brier"], expected)
        self.assertAlmostEqual(sum(result["decomposition"].values()), expected)

    def test_only_projected_columns_and_never_traces_or_assets(self):
        opened = []
        original = pq.ParquetFile

        class Reader:
            def __init__(self, path):
                self.path = Path(path)
                self.reader = original(path)

            def iter_batches(self, *, batch_size, columns):
                opened.append((self.path.name, columns))
                self_outer.assertTrue(columns)
                self_outer.assertNotIn("grading_criterion", columns)
                self_outer.assertNotIn("test_condition", columns)
                return self.reader.iter_batches(batch_size=batch_size, columns=columns)

        self_outer = self
        with patch.object(sim.pq, "ParquetFile", Reader):
            sim.run_simulation(self.data, lambda x, y: 0.5, max_sample=80)
        self.assertEqual({name for name, _ in opened},
                         {"subjects.parquet", "items.parquet", "responses.parquet", "benchmarks.parquet"})

    def test_nonbinary_and_aggregate_benchmarks_are_skipped_before_data_reads(self):
        data = self.work / "filter-data"
        build_data(data)
        for index, response_type, granularity in ((1, "fraction", "item"), (2, "binary", "aggregate")):
            bid = f"bench-{index}"
            folder = data / bid / "formatted_tables"
            write_table(folder, "benchmarks", [{"benchmark_id": bid, "response_type": response_type,
                                               "granularity": granularity}])
            # If excluded response files are opened, pyarrow will raise.
            (folder / "responses.parquet").write_bytes(b"NOT PARQUET")
        result = sim.run_simulation(data, lambda x, y: 0.5, max_sample=80)
        self.assertEqual(result["eligible_pair_count"], 2)
        excluded = {b["original_benchmark_id"]: b["excluded_reason"]
                    for d in result["inventory"] for b in d["benchmarks"]}
        self.assertEqual(excluded, {"bench-0": None, "bench-1": "requires_binary_item",
                                    "bench-2": "requires_binary_item"})

    def test_strategies_cap_and_pre_cap_eligibility(self):
        balanced = self.constant["runs"][0]
        packed = sim.run_simulation(self.data, lambda x, y: 0.5, pairs_strategy="packed",
                                   max_sample=240, seed=51)["runs"][0]
        self.assertEqual([p["sampled_unique_items"] for p in balanced["pairs"]], [80, 80, 80])
        self.assertEqual(sorted(p["sampled_unique_items"] for p in packed["pairs"]), [101, 101])
        for strategy in ("balanced", "packed"):
            result = sim.run_simulation(self.data, lambda x, y: 0.5, pairs_strategy=strategy)
            self.assertEqual(result["eligible_pair_count"], 6)
            self.assertLessEqual(result["runs"][0]["sampled_unique_subject_items"], 1000)
            for pair in result["runs"][0]["pairs"]:
                self.assertGreaterEqual(pair["sampled_unique_items"], 80)
                self.assertNotIn(pair["subject_id"], {sim.anonymous("subject", "old"),
                                                      sim.anonymous("subject", "short")})

    def test_benchmark_split_is_stable_disjoint_and_seed_independent(self):
        assignments = {f"bench-{i}": benchmark_split(f"bench-{i}", 0.5) for i in range(3)}
        self.assertEqual(set(assignments.values()), {"tune", "holdout"})
        observed = {}
        for split in ("tune", "holdout"):
            result = sim.run_simulation(self.data, lambda x, y: 0.5, split=split,
                                       holdout_fraction=0.5, max_sample=80)
            ids = {b["original_benchmark_id"] for d in result["inventory"] for b in d["benchmarks"]
                   if b["excluded_reason"] is None}
            self.assertEqual(ids, {k for k, v in assignments.items() if v == split})
            observed[split] = ids
        self.assertFalse(observed["tune"] & observed["holdout"])
        for name in ("a", "b", "bench-0"):
            self.assertEqual(benchmark_split(name, 0), "tune")
            self.assertEqual(benchmark_split(name, 1), "holdout")
            expected = "holdout" if int(hashlib.sha256(name.encode()).hexdigest(), 16) < 2**255 else "tune"
            self.assertEqual(benchmark_split(name, 0.5), expected)

    def test_reproducibility_and_run_summary_std(self):
        a = sim.run_simulation(self.data, lambda x, y: 0.7, runs=2, max_sample=80, seed=25)
        b = sim.run_simulation(self.data, lambda x, y: 0.7, runs=2, max_sample=80, seed=25)
        self.assertEqual(a, b)
        self.assertEqual(a["summary"]["alc_std"], statistics.pstdev(r["alc"] for r in a["runs"]))
        self.assertEqual([r["seed"] for r in a["runs"]], [25, 26])
        self.assertNotEqual(a["runs"][0]["pairs"][0]["evaluation_item_ids"],
                            a["runs"][1]["pairs"][0]["evaluation_item_ids"])

    def test_hook_signatures_prediction_only_when_needed_and_no_retry(self):
        contexts = []

        def subset(input, *, context):
            contexts.append(context)
            return True

        result = sim.run_simulation(self.data, lambda x, y: 0.5, subset, max_sample=80)
        self.assertEqual(result["runs"][0]["acquisition_prediction_calls"], 0)
        self.assertEqual(len(contexts), 31)
        captured = []

        def positional(a, b, c, d, /):
            captured.append((a, b, c, d))
            return True

        hook = sim.AcquisitionHook(positional)
        self.assertTrue(hook.needs_prediction)
        self.assertTrue(hook([{}, {}], 0.5, [], {}))
        self.assertEqual(captured, [([{}, {}], 0.5, [], {})])
        count = []

        def broken(input, context=None):
            count.append(1)
            raise TypeError("inside participant code")

        with self.assertRaisesRegex(TypeError, "inside participant"):
            sim.AcquisitionHook(broken)([{}, {}], None, [], {})
        self.assertEqual(len(count), 1)
        with self.assertRaisesRegex(ValueError, "legacy ranking"):
            sim.AcquisitionHook(lambda input: 1.0)
        with self.assertRaisesRegex(ValueError, "bool"):
            sim.AcquisitionHook(lambda input, context: 1)([{}, {}], None, [], {})

    def test_hook_exhaustion_is_reported_without_forced_reveals(self):
        result = sim.run_simulation(self.data, lambda x, y: 0.5,
                                   lambda input, context: False, max_sample=80)
        pair = result["runs"][0]["pairs"][0]
        cp = pair["checkpoints"]["31"]
        self.assertEqual(cp["labels_acquired"], 0)
        self.assertTrue(cp["acquisition_exhausted"])
        self.assertFalse(cp["budget_reached"])
        self.assertEqual(result["runs"][0]["acquisition_calls"], 40)

    def test_invalid_probabilities_stop_and_nonstrings_are_json(self):
        for value in (float("nan"), float("inf"), -0.01, 1.01, None):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                sim.run_simulation(self.data, lambda x, y: value, max_sample=80)
        self.assertEqual(sim.text_value(None), "")
        self.assertEqual(sim.text_value("abc"), "abc")
        self.assertEqual(json.loads(sim.text_value({"a": [True, 2]})), {"a": [True, 2]})
        self.assertEqual(sim.anonymous("subject", "s0"),
                         "subject_" + hashlib.sha256(b"s0").hexdigest())

    def test_predictions_reused_per_checkpoint_but_not_across_budgets(self):
        # Distinct item IDs may have identical full inputs. Scoring keeps every row.
        a = [{"who": "s"}, {"q": "same"}]
        pair = sim.Pair(self.data / "bench-0" / "formatted_tables", "s0", "bench-0", 80)
        state = sim.PairState(pair, [], [("i0", a, [0, 1, 1]), ("i1", a, [1])], ["i0", "i1"], 2)
        calls = []

        def predictor(input, labeled):
            calls.append(1)
            return 0.5

        run = sim.simulate_run([state], predictor, sim.AcquisitionHook(None), 0)
        self.assertEqual(len(calls), 6)
        self.assertEqual(run["evaluation_prediction_calls_by_budget"], {str(b): 1 for b in sim.BUDGETS})
        self.assertEqual(run["pairs"][0]["checkpoints"]["31"]["scored_response_rows"], 4)

    def test_e1_model_end_to_end_cli_with_optional_labeling(self):
        labeling = self.work / "labeling.py"
        labeling.write_text("def acquisition_function(input, *, context):\n    return True\n")
        for hook_args, mode in (([], "random"), (["--labeling", str(labeling)], "stream")):
            with self.subTest(mode=mode):
                out = self.work / f"e1-{mode}.json"
                command = [sys.executable, "-B", "f1/simulate.py", "--data", str(self.data),
                           "--model", "submission_e1/model.py", "--benchmarks", "bench-0,bench-1,bench-2",
                           "--runs", "2", "--seed", "21", "--pairs-strategy", "packed",
                           "--max-sample", "240", *hook_args, "--out", str(out)]
                completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                result = json.loads(out.read_text())
                self.assertEqual(result["conditions"]["acquisition_mode"], mode)
                self.assertEqual(len(result["runs"]), 2)
                self.assertTrue(math.isfinite(result["summary"]["alc_mean"]))
                self.assertTrue(0 <= result["summary"]["alc_mean"] <= 1)
                self.assertEqual(result["runs"][0]["brier_by_budget"]["0"], 0.25)
                self.assertNotEqual(result["runs"][0]["brier_by_budget"]["31"], 0.25)


if __name__ == "__main__":
    unittest.main()
