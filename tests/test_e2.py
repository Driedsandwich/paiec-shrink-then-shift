"""E2 contracts, using synthetic text only. No dataset text is displayed."""

import copy
import importlib.util
import math
from pathlib import Path
import random
import unittest
from unittest.mock import patch

from variants import v2
from variants import e2_features as e2

ROOT = Path(__file__).resolve().parents[1]
SUBJECT = {"normalized_name": "synthetic", "release_date": "2025-01-01"}


def item(text="Synthetic task", bid="bench", **extra):
    return dict(item_content=text, benchmark_id=bid, item_features="{}", interactors="", **extra)


def params():
    return {"weights": [0.1] * len(e2.FEATURE_NAMES), "means": [0.0] * len(e2.FEATURE_NAMES),
            "scales": [1.0] * len(e2.FEATURE_NAMES), "intercept": 0.0, "global_mean": 0.0}


def submitted():
    spec = importlib.util.spec_from_file_location("e2_submission_test", ROOT / "submission_e2/model.py")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    return model


class FeatureTests(unittest.TestCase):
    def test_malformed_ordinary_inputs_are_finite(self):
        values = [None, {}, [], 4, "invalid", {"item_content": None},
                  {"item_content": 9, "item_features": "{broken", "interactors": []},
                  {"item_content": "\ud800", "item_features": {"x": float("nan")}}]
        for value in values:
            for subject in (None, {}, [], {"release_date": None}):
                vector = e2.feature_vector(value, subject)
                self.assertEqual(len(vector), len(e2.FEATURE_NAMES))
                self.assertTrue(all(math.isfinite(v) for v in vector))

    def test_empty_body_has_missing_year_and_zero_length(self):
        f = e2.extract_features(item(""), SUBJECT)
        self.assertEqual(f["log_chars"], 0)
        self.assertEqual(f["log_estimated_lines"], 0)
        self.assertEqual(f["no_year"], 1)
        self.assertEqual(f["year_minus_release"], 0)

    def test_structural_features(self):
        text = "2023\nA) alpha\n(B) beta\n```python\ndef demo(): import x; {}\n$ \\frac\nhttps://example.invalid"
        f = e2.extract_features(dict(item(text), item_features='{"x":1,"y":2}', interactors='["agent"]'), SUBJECT)
        for name in ("log_code_markers", "log_math_markers", "log_choice_markers", "has_url", "has_interactors"):
            self.assertGreater(f[name], 0)
        self.assertEqual(f["feature_key_count"], 2)
        self.assertEqual(f["year_minus_release"], -2)
        self.assertEqual(f["no_year"], 0)
        self.assertAlmostEqual(f["log_chars"], math.log1p(len(text)))

    def test_year_uses_maximum_and_separate_missing_flags(self):
        f = e2.extract_features(item("1999 2027 2023"), SUBJECT)
        self.assertEqual(f["year_minus_release"], 2)
        f = e2.extract_features(item("2027"), {"release_date": ""})
        self.assertEqual(f["no_release_year"], 1)
        self.assertEqual(f["no_year"], 0)
        self.assertEqual(f["year_minus_release"], 0)

    def test_huge_body_uses_bounded_scan_but_full_length(self):
        text = "x" * 5000000 + "\n2027"
        f = e2.extract_features(item(text), SUBJECT)
        self.assertAlmostEqual(f["log_chars"], math.log1p(len(text)))
        self.assertEqual(f["year_minus_release"], 2)
        self.assertTrue(all(math.isfinite(v) for v in f.values()))

    def test_empty_json_interactors_absent(self):
        for value in (None, "", "null", "[]", "{}", '""', [], {}):
            self.assertEqual(e2.extract_features(dict(item(), interactors=value), SUBJECT)["has_interactors"], 0)

    def test_metadata_is_not_executed(self):
        self.assertEqual(e2.extract_features(dict(item(), item_features="__import__('os').system('false')"), SUBJECT)["feature_key_count"], 0)

    def test_benchmark_and_subject_name_do_not_change_features(self):
        self.assertEqual(e2.feature_vector(item(bid="one"), SUBJECT),
                         e2.feature_vector(item(bid="two"), dict(SUBJECT, normalized_name="other")))


class PredictionTests(unittest.TestCase):
    def test_gamma_zero_exactly_matches_variants_and_submission_e1b(self):
        spec = importlib.util.spec_from_file_location("e1b_reference", ROOT / "submission_e1b/model.py")
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        rng = random.Random(53)
        labels = []
        for n in range(90):
            query = [SUBJECT, item(bid="bench" if n % 3 else "other")]
            for function in (reference.predict, lambda x, y: v2(x, y, k=4, b=4, prior_mean=e2.PRIOR_MEAN)):
                self.assertEqual(e2.predict_with_parameters(query, labels, params(), 0), function(query, labels))
            subject = SUBJECT if n % 2 else dict(SUBJECT, normalized_name="another")
            labels.append([[subject, item(str(n), "bench" if n % 4 else "other")], rng.randrange(2)])

    def test_gamma_zero_skips_feature_extraction(self):
        with patch.object(e2, "feature_score", side_effect=RuntimeError("synthetic failure")):
            self.assertEqual(e2.predict_with_parameters([SUBJECT, item()], [], {}, 0), e2.PRIOR_MEAN)

    def test_fallback_is_e1b(self):
        query = [SUBJECT, item()]
        labels = [[[SUBJECT, item()], 1]]
        self.assertEqual(e2.predict_with_parameters(query, labels, {}, .5), e2.e1b(query, labels))
        with patch.object(e2, "feature_score", side_effect=ValueError("synthetic failure")):
            self.assertEqual(e2.predict_with_parameters(query, labels, params(), 1), e2.e1b(query, labels))

    def test_labeled_center_uses_same_benchmark_only(self):
        p = params()
        query = [SUBJECT, item("same")]
        labels = [[[SUBJECT, item("same")], 1], [[SUBJECT, item("z" * 500, "other")], 0]]
        self.assertAlmostEqual(e2.predict_with_parameters(query, labels, p, 1), e2.e1b(query, labels))

    def test_global_mean_used_when_no_same_benchmark_label(self):
        p = params()
        query = [SUBJECT, item()]
        score = e2.feature_score(query[1], query[0], p)
        expected = e2.sigmoid(math.log(e2.PRIOR_MEAN / (1-e2.PRIOR_MEAN)) + .5 * score)
        self.assertAlmostEqual(e2.predict_with_parameters(query, [], p, .5), expected)

    def test_range_and_malformed_inputs(self):
        model = submitted()
        for query in (None, [], [None, None], [SUBJECT, item("")], [SUBJECT, item("x" * 5000000)]):
            for labels in (None, [], [None, "broken", [[SUBJECT, item()], 2]], 5):
                for function in (model.predict, lambda x,y: e2.predict_with_parameters(x,y,params(),1)):
                    result = function(query, labels)
                    self.assertTrue(math.isfinite(result) and .001 <= result <= .999)

    def test_extreme_logits_are_clipped(self):
        p = params()
        p["weights"][0] = 1e10
        query = [SUBJECT, item("x" * 5000)]
        self.assertEqual(e2.predict_with_parameters(query, [], p, 1), .999)
        p["weights"][0] = -1e10
        self.assertEqual(e2.predict_with_parameters(query, [], p, 1), .001)

    def test_no_input_mutation_or_call_state(self):
        model = submitted()
        query = [SUBJECT, item()]
        labels = [[[SUBJECT, item("observed")], 1]]
        before = copy.deepcopy((query, labels))
        first = model.predict(query, labels)
        model.predict([SUBJECT, item("other")], [])
        self.assertEqual(first, model.predict(query, labels))
        self.assertEqual(before, (query, labels))

    def test_embedded_model_matches_local_implementation(self):
        model = submitted()
        for text in ("", "A) x\nB) y\n2026", "x" * 1000000):
            query = [SUBJECT, item(text)]
            labels = [[[dict(SUBJECT, normalized_name="other"), item("2023")], 0]]
            self.assertEqual(model.predict(query, labels),
                             e2.predict_with_parameters(query, labels, model.PARAMETERS, model.GAMMA))

    def test_score_cache_exists_only_within_one_call(self):
        query = [SUBJECT, item("identical")]
        labels = [[[dict(SUBJECT, normalized_name=str(i)), item("identical")], i % 2] for i in range(31)]
        with patch.object(e2, "feature_score", wraps=e2.feature_score) as score:
            first = e2.predict_with_parameters(query, labels, params(), 1)
            self.assertEqual(score.call_count, 1)
            second = e2.predict_with_parameters(query, labels, params(), 1)
            self.assertEqual(score.call_count, 2)
            self.assertEqual(first, second)

    def test_cache_distinguishes_length_release_and_metadata(self):
        query = [SUBJECT, item("2026")]
        labels = [
            [[dict(SUBJECT, release_date="2026-01"), item("2026")], 0],
            [[SUBJECT, dict(item("2026"), item_features='{"extra":1}')], 1],
            [[SUBJECT, dict(item("2026"), interactors='["agent"]')], 0],
            [[SUBJECT, item("2026 extra")], 1],
        ]
        p = params()
        center = sum(e2.feature_score(x[1], x[0], p) for x, _ in labels) / len(labels)
        base = e2.e1b(query, labels)
        expected = e2.sigmoid(math.log(base/(1-base)) + e2.feature_score(query[1],query[0],p) - center)
        self.assertAlmostEqual(e2.predict_with_parameters(query, labels, p, 1), expected)


class TrainingTests(unittest.TestCase):
    def test_rank_ties_use_average_rank(self):
        from tools.train_e2 import rank_values
        self.assertEqual(rank_values([3, 1, 1, 9]).tolist(), [2, .5, .5, 3])

    def test_ridge_fit_is_finite_with_constant_features(self):
        import numpy as np
        from tools.train_e2 import fit, predict_numeric
        x = np.array([[0., 1.], [2., 1.], [4., 1.]])
        p = fit(x, np.array([-1., 0., 1.]), np.ones(3), .1)
        self.assertEqual(p["scales"][1], 1)
        self.assertTrue(np.isfinite(predict_numeric(x, p)).all())
        self.assertAlmostEqual(p["global_mean"], 0)


if __name__ == "__main__":
    unittest.main()
