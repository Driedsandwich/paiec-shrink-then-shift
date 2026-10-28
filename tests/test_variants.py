"""Hand-calculated formulas, defensive contracts and content-free replay tests."""

import copy
from functools import partial
import importlib.util
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from f1 import simulate as sim
from tools import tune_e1 as tune
import variants as v

ROOT = Path(__file__).resolve().parents[1]


def input_for(subject="a", benchmark="x"):
    return [{"normalized_name": subject}, {"benchmark_id": benchmark}]


def evidence(subject, benchmark, ys):
    return [[input_for(subject, benchmark), y] for y in ys]


class VariantTests(unittest.TestCase):
    def test_v1_hand_formula(self):
        self.assertAlmostEqual(v.v1(input_for(), evidence("a", "x", [1, 1, 0]),
                                   k=2, prior_mean=0.4), 2.8 / 5)

    def test_v2_hand_formula_and_own_count_once(self):
        labels = evidence("a", "x", [1]) + evidence("b", "x", [0, 0])
        # Other rate=(0+2*.5)/(2+2)=.25; own=(1+2*.25)/(1+2)=.5.
        self.assertAlmostEqual(v.v2(input_for(), labels, k=2, b=2), 0.5)

    def test_own_labels_only_enter_once_in_all_variants(self):
        labels = evidence("a", "x", [1] * 3)
        for variant in ("V1", "V2", "V3"):
            self.assertAlmostEqual(v.predict(input_for(), labels, variant=variant,
                                             k=2, b=8, a=32), 4 / 5)

    def test_v3_hand_logit_offset(self):
        labels = (evidence("a", "x", [1, 0]) + evidence("b", "x", [1, 0]) +
                  evidence("a", "y", [1, 1, 1, 0]) + evidence("b", "y", [1, 0, 0, 0]))
        # delta=4*(log(3)-log(1/3))/(4+4)=log(3).
        # benchmark prior=.5 -> adjusted=.75 -> own shrinkage=(1+2*.75)/4.
        self.assertAlmostEqual(v.v3(input_for(), labels, k=2, b=2, a=4), 0.625)

    def test_v3_weighting_multiple_other_benchmarks(self):
        labels = (evidence("a", "y", [1,1,1,0]) + evidence("b", "y", [1,0,0,0]) +
                  evidence("a", "z", [1,0]) + evidence("b", "z", [1,1,1,0]))
        expected_delta = (4 * 2 * math.log(3) - 2 * math.log(3)) / (6 + 2)
        self.assertAlmostEqual(v.v3(input_for(), labels, a=2),
                               1 / (1 + math.exp(-expected_delta)))

    def test_v3_missing_comparator_skipped(self):
        labels = evidence("b", "x", [1, 1, 0]) + evidence("a", "y", [1]*9)
        self.assertEqual(v.v2(input_for(), labels), v.v3(input_for(), labels))

    def test_target_evidence_does_not_create_ability_offset(self):
        labels = evidence("a", "x", [1]*9) + evidence("b", "x", [0]*9)
        self.assertEqual(v.v2(input_for(), labels), v.v3(input_for(), labels))

    def test_v1_ignores_all_other_pairs(self):
        labels = evidence("b", "x", [1]*20) + evidence("a", "z", [0]*20)
        self.assertEqual(v.v1(input_for(), labels, prior_mean=0.4), 0.4)

    def test_v3_ability_shrinks_toward_zero(self):
        labels = evidence("a", "y", [1]*4) + evidence("b", "y", [0]*4)
        self.assertGreater(v.v3(input_for(), labels, a=2), v.v3(input_for(), labels, a=32))

    def test_baselines_unsmoothed(self):
        self.assertEqual(v.constant(None, None), 0.5)
        self.assertEqual(v.empirical(input_for(), []), 0.5)
        self.assertEqual(v.empirical(input_for(), evidence("a", "x", [1])), 1)
        self.assertEqual(v.empirical(input_for(), evidence("a", "x", [0])), 0)
        self.assertEqual(v.empirical(input_for(), evidence("a", "x", [1,0,0])), 1/3)

    def test_malformed_inputs_and_parameters_never_raise(self):
        bad_inputs = [None, 1, "x", [], [1], [1, None], [1, {}], [1, {"benchmark_id": []}]]
        for variant in ("V1", "V2", "V3", "empirical", "constant"):
            for input in bad_inputs + [input_for()]:
                for labels in (None, 1, "x", {}, [None, 1, [None, 2], [[None, None], 1]]):
                    p = v.predict(input, labels, variant=variant, k=None, b=-1,
                                  a=float("inf"), prior_mean=float("nan"))
                    self.assertTrue(math.isfinite(p) and 0 <= p <= 1)

    def test_range_extremes(self):
        labels = evidence("a", "y", [1]*31) + evidence("b", "y", [0]*31)
        for variant in ("V1", "V2", "V3"):
            for m in (-10, 0, 0.5, 1, 20):
                for y in (0,1):
                    p = v.predict(input_for(), labels + evidence("a", "x", [y]*31),
                                  variant=variant, k=1e-10, prior_mean=m)
                    self.assertTrue(0.001 <= p <= 0.999)

    def test_bad_entries_skipped(self):
        labels = [None, [input_for(), 2], [[{}, None], 1], [input_for(), float("nan")]]
        self.assertEqual(v.v2(input_for(), labels + evidence("a", "x", [1])), 0.6)

    def test_no_mutation_or_state(self):
        input, labels = input_for(), evidence("a", "x", [1])
        before = copy.deepcopy((input, labels))
        for function in (v.v1, v.v2, v.v3):
            p = function(input, labels)
            function(input_for("other"), evidence("b", "x", [0]*31))
            self.assertEqual(function(input, labels), p)
        self.assertEqual((input, labels), before)

    def test_v2_matches_current_e1(self):
        spec = importlib.util.spec_from_file_location("original_e1", ROOT / "submission_e1/model.py")
        model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model)
        rng = random.Random(42)
        labels = []
        for _ in range(80):
            labels += evidence(rng.choice("abc"), rng.choice("xy"), [rng.randrange(2)])
            self.assertAlmostEqual(v.v2(input_for(), labels), model.predict(input_for(), labels))

    def test_content_projection_never_requests_content(self):
        requested = []

        class Batch:
            def to_pylist(self):
                return [{"item_id": "i", "benchmark_id": "b"}]

        class FakeParquet:
            def __init__(self, path):
                pass

            def iter_batches(self, **kwargs):
                requested.extend(kwargs["columns"])
                return iter([Batch()])

        with patch.object(tune.pq, "ParquetFile", FakeParquet):
            rows = list(tune.safe_rows(Path("unused"), "items", ("item_id", "benchmark_id", "content")))
        self.assertNotIn("content", requested)
        self.assertEqual(rows[0]["content"], "")

    def test_paired_difference_standard_error(self):
        a = [{"seed": i, "alc": x} for i,x in enumerate([0.1,0.3,0.5])]
        b = [{"seed": i, "alc": 0.2} for i in range(3)]
        d = tune.paired(a,b)
        self.assertAlmostEqual(d["mean"], 0.1)
        self.assertAlmostEqual(d["standard_error"], 0.2/math.sqrt(3))

    def test_synthetic_replay_matches_full_f1_every_budget(self):
        # F1's fixture builder writes only synthetic questions in the allowed tree.
        from test_f1 import build_data
        tune.OUT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tune.OUT) as tmp:
            data = Path(tmp)
            build_data(data)
            with tune.content_free_f1():
                pairs, _ = sim.inventory(data, None, "all", 0.3, tune.YEARS)
                for strategy in ("balanced", "packed"):
                    plan = tune.make_plan(pairs, 7, strategy, max_sample=240)
                    for variant in ("constant", "empirical", "V1", "V2", "V3"):
                        config = dict(variant=variant, k=2, b=4, a=8, prior_mean=0.4)
                        fast = tune.score_plan(plan, config)
                        reference = sim.run_simulation(data, partial(v.predict, **config), seed=7,
                            pairs_strategy=strategy, max_sample=240)["runs"][0]
                        for budget in sim.BUDGETS:
                            self.assertAlmostEqual(fast["brier_by_budget"][str(budget)],
                                                   reference["brier_by_budget"][str(budget)], places=13)


if __name__ == "__main__":
    unittest.main()
