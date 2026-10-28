import importlib.util
import math
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "e1_model", Path(__file__).resolve().parents[1] / "submission_e1" / "model.py")
model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model)

A = {"normalized_name": "a", "provider": "p", "release_date": "", "access_date": "",
     "harness": "", "harness_version": "", "reasoning_effort": "", "subject_features_extra": ""}
B = dict(A, normalized_name="b")


def item(bench, text="q"):
    return {"item_content": text, "item_features": "", "interactors": "", "benchmark_id": bench}


class E1Test(unittest.TestCase):
    def test_budget_zero_is_prior(self):
        self.assertEqual(model.predict([A, item("x")], []), model.PRIOR_MEAN)
        self.assertEqual(model.predict([A, item("x")], None), model.PRIOR_MEAN)

    def test_missing_benchmark_falls_back(self):
        labeled = [[[A, item("x")], 1]]
        self.assertEqual(model.predict([A, item("")], labeled), model.PRIOR_MEAN)

    def test_single_failure_is_not_zero(self):
        p = model.predict([A, item("x")], [[[A, item("x")], 0]])
        self.assertTrue(0.2 < p < 0.5, p)

    def test_other_subject_labels_move_benchmark_rate(self):
        others = [[[B, item("x", str(i))], 1] for i in range(10)]
        p = model.predict([A, item("x")], others)
        self.assertGreater(p, 0.7)

    def test_other_benchmark_ignored(self):
        labeled = [[[A, item("y")], 1]] * 5
        self.assertEqual(model.predict([A, item("x")], labeled), model.PRIOR_MEAN)

    def test_own_labels_dominate_with_more_evidence(self):
        others = [[[B, item("x", str(i))], 1] for i in range(10)]
        own = [[[A, item("x", "o" + str(i))], 0] for i in range(31)]
        self.assertLess(model.predict([A, item("x")], others + own), 0.25)

    def test_own_labels_counted_once(self):
        own = [[[A, item("x", "o" + str(i))], 1] for i in range(31)]
        expected = (31 + model.PAIR_STRENGTH * model.PRIOR_MEAN) / (31 + model.PAIR_STRENGTH)
        self.assertAlmostEqual(model.predict([A, item("x")], own), expected)

    def test_bad_entries_skipped_and_output_finite(self):
        labeled = [None, [[A, item("x")], 2], [[A, "not-a-dict"], 1], [[A, item("x")], 1]]
        p = model.predict([A, item("x")], labeled)
        self.assertTrue(math.isfinite(p) and 0.0 < p < 1.0)


if __name__ == "__main__":
    unittest.main()
