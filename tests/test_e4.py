import importlib.util
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name + "_model", ROOT / name / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model = load("submission_e4")
e1 = load("submission_e1")

A = {"normalized_name": "a", "provider": "p", "release_date": "", "access_date": "",
     "harness": "", "harness_version": "", "reasoning_effort": "", "subject_features_extra": ""}
B = dict(A, normalized_name="b")


def item(bench, text="q"):
    return {"item_content": text, "item_features": "", "interactors": "", "benchmark_id": bench}


class E4Test(unittest.TestCase):
    def test_budget_zero_is_half(self):
        self.assertEqual(model.PRIOR_MEAN, 0.5)
        self.assertEqual(model.predict([A, item("x")], []), 0.5)
        self.assertEqual(model.predict([A, item("x")], None), 0.5)

    def test_missing_benchmark_falls_back(self):
        labeled = [[[A, item("x")], 1]]
        self.assertEqual(model.predict([A, item("")], labeled), 0.5)

    def test_malformed_input_falls_back(self):
        labeled = [[[A, item("x")], 1]]
        self.assertEqual(model.predict(None, labeled), 0.5)
        self.assertEqual(model.predict([A, "not-a-dict"], labeled), 0.5)

    def test_equals_e1_without_other_benchmark_labels(self):
        labeled = ([[[B, item("x", str(i))], i % 2] for i in range(7)]
                   + [[[A, item("x", "o" + str(i))], 0] for i in range(3)])
        for evidence in ([], labeled[:1], labeled[:7], labeled):
            self.assertAlmostEqual(model.predict([A, item("x")], evidence),
                                   e1.predict([A, item("x")], evidence))

    def test_other_benchmark_labels_move_global_rate(self):
        others = [[[B, item("y", str(i))], 0] for i in range(20)]
        expected_global = (0 + model.GLOBAL_STRENGTH * 0.5) / (20 + model.GLOBAL_STRENGTH)
        self.assertAlmostEqual(model.predict([A, item("x")], others), expected_global)
        self.assertLess(model.predict([A, item("x")], others), 0.5)

    def test_each_label_enters_one_level(self):
        own = [[[A, item("x", "o" + str(i))], 1] for i in range(5)]
        same_bench = [[[B, item("x", "b" + str(i))], 0] for i in range(6)]
        other_bench = [[[A, item("y", "g" + str(i))], 1] for i in range(8)]
        g = (8 + model.GLOBAL_STRENGTH * 0.5) / (8 + model.GLOBAL_STRENGTH)
        b = (0 + model.BENCH_STRENGTH * g) / (6 + model.BENCH_STRENGTH)
        p = (5 + model.PAIR_STRENGTH * b) / (5 + model.PAIR_STRENGTH)
        self.assertAlmostEqual(model.predict([A, item("x")], own + same_bench + other_bench), p)

    def test_own_labels_dominate_with_more_evidence(self):
        others = [[[B, item("y", str(i))], 1] for i in range(50)]
        own = [[[A, item("x", "o" + str(i))], 0] for i in range(31)]
        self.assertLess(model.predict([A, item("x")], others + own), 0.25)

    def test_bad_entries_skipped_and_output_finite(self):
        labeled = [None, [[A, item("x")], 2], [[A, "not-a-dict"], 1], [[A, {"benchmark_id": 3}], 1],
                   [[A, item("x")], 1], [[B, item("y")], 0]]
        p = model.predict([A, item("x")], labeled)
        self.assertTrue(math.isfinite(p) and 0.0 < p < 1.0)


if __name__ == "__main__":
    unittest.main()
