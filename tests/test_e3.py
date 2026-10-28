"""E3 contracts with synthetic inputs only; no measurement-db text is used."""

import copy
import importlib.util
import math
from pathlib import Path
import random
import unittest
from unittest.mock import patch

from submission_e2 import model as e2
from variants import e3_same_item as e3

ROOT = Path(__file__).resolve().parents[1]
SUBJECT = {"normalized_name": "synthetic-target", "release_date": "2025-01-01"}
OTHER = dict(SUBJECT, normalized_name="synthetic-other")


def item(text="Synthetic item", benchmark="bench"):
    return {"item_content": text, "item_features": "{}", "interactors": "", "benchmark_id": benchmark}


def submission():
    spec = importlib.util.spec_from_file_location("e3_submission_test", ROOT / "submission_e3/model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SameItemTests(unittest.TestCase):
    def test_no_labels_exactly_e2(self):
        for text in ("", "Synthetic 2026 task", "x" * 100000):
            query = [SUBJECT, item(text)]
            for mode in ("probability", "logit"):
                self.assertEqual(e3.predict(query, [], mode=mode), e2.predict(query, []))

    def test_other_items_exactly_e2(self):
        query = [SUBJECT, item()]
        labels = [[[OTHER, item("Synthetic other item")], 1]]
        self.assertEqual(e3.same_item_stats(query, labels), (0, 0.0, 0.0))
        self.assertEqual(e3.predict(query, labels), e2.predict(query, labels))

    def test_own_subject_never_contributes_residual(self):
        query = [SUBJECT, item()]
        for y in (0, 1):
            labels = [[[copy.deepcopy(SUBJECT), item()], y]]
            self.assertEqual(e3.same_item_stats(query, labels), (0, 0.0, 0.0))
            for mode in ("probability", "logit"):
                self.assertEqual(e3.predict(query, labels, mode=mode), e2.predict(query, labels))

    def test_own_subject_excluded_in_mixed_matching_entries(self):
        query = [SUBJECT, item()]
        labels = [[query, 0], [[OTHER, item()], 1]]
        n, residual, logit_residual = e3.same_item_stats(query, labels)
        q = e2.e1b([OTHER, item()], labels)
        self.assertEqual(n, 1)
        self.assertEqual(residual, 1-q)
        self.assertEqual(logit_residual, (1-q)/max(q*(1-q), .05))

    def test_each_of_four_fields_must_match_exactly(self):
        query = [SUBJECT, item()]
        for field in e3.ITEM_FIELDS:
            changed = item()
            changed[field] += "x"
            labels = [[[OTHER, changed], 1]]
            self.assertEqual(e3.same_item_stats(query, labels)[0], 0, field)
            self.assertEqual(e3.predict(query, labels), e2.predict(query, labels), field)

    def test_missing_field_is_not_a_match(self):
        for field in e3.ITEM_FIELDS:
            query_item = item()
            del query_item[field]
            labels = [[[OTHER, copy.deepcopy(query_item)], 1]]
            self.assertEqual(e3.same_item_stats([SUBJECT, query_item], labels)[0], 0)

    def test_nested_metadata_compared_by_dictionary_equality(self):
        observed = dict(item(), item_features={"a": [1, {"x": 2}], "b": 3}, interactors=[{"agent": "synthetic"}])
        query_item = dict(observed, item_features={"b": 3, "a": [1, {"x": 2}]})
        self.assertEqual(e3.same_item_stats([SUBJECT, query_item], [[[OTHER, observed], 0]])[0], 1)

    def test_full_content_is_compared_beyond_e2_scan(self):
        a = "x"*3000 + "A" + "x"*3000
        b = "x"*3000 + "B" + "x"*3000
        self.assertEqual(e3.same_item_stats([SUBJECT, item(a)], [[[OTHER, item(b)], 1]])[0], 0)

    def test_q_t_is_e1b_with_matching_label_included(self):
        query = [SUBJECT, item()]
        labels = [[[OTHER, item()], 1], [[OTHER, item("Synthetic second")], 0],
                  [[dict(OTHER, normalized_name="third"), item("Synthetic third")], 1]]
        n, residual, _ = e3.same_item_stats(query, labels)
        self.assertEqual(n, 1)
        self.assertEqual(residual, 1-e2.e1b([OTHER, item()], labels))

    def test_probability_and_logit_formulas(self):
        x = [SUBJECT, item()]
        labels = [[[OTHER, item()], 1]]
        baseline = e2.predict(x, labels)
        n, residual, lr = e3.same_item_stats(x, labels)
        self.assertEqual(e3.predict(x, labels, shrinkage=2, eta=.5, mode="probability"),
                         e2.clip(baseline+.5*residual/(n+2)))
        expected = e2.clip(e2.sigmoid(math.log(baseline/(1-baseline))+.5*lr/(n+2)))
        self.assertEqual(e3.predict(x, labels, shrinkage=2, eta=.5, mode="logit"), expected)

    def test_invalid_settings_return_exact_e2(self):
        x = [SUBJECT, item()]
        labels = [[[OTHER, item()], 0]]
        for config in ({"shrinkage": 0}, {"shrinkage": -1}, {"shrinkage": "broken"},
                       {"shrinkage": float("nan")}, {"eta": float("inf")}, {"mode": "invalid"}):
            self.assertEqual(e3.predict(x, labels, **config), e2.predict(x, labels))

    def test_internal_exception_returns_exact_e2(self):
        x = [SUBJECT, item()]
        labels = [[[OTHER, item()], 1]]
        with patch.object(e3, "same_item_stats", side_effect=ValueError("synthetic failure")):
            self.assertEqual(e3.predict(x, labels), e2.predict(x, labels))

    def test_broken_labeled_rows_are_skipped(self):
        x = [SUBJECT, item()]
        valid = [[OTHER, item()], 1]
        labels = [None, 2, "broken", [], [1, 2, 3], [[None, None], 1],
                  [[OTHER, item()], 2], [[OTHER, item()], float("nan")],
                  [[OTHER, item()], "1"], valid]
        self.assertEqual(e3.same_item_stats(x, labels)[0], 1)
        for mode in ("probability", "logit"):
            self.assertTrue(math.isfinite(e3.predict(x, labels, mode=mode)))

    def test_malformed_inputs_never_raise_and_stay_in_range(self):
        module = submission()
        queries = (None, [], 5, "broken", [None, None], [SUBJECT, None], [None, item()],
                   [SUBJECT, {}], [SUBJECT, item()], [SUBJECT, dict(item(), item_content="\ud800")])
        evidence = (None, 5, "broken", {}, [], [None, "broken", [[OTHER, item()], 2]])
        for query in queries:
            for labels in evidence:
                for predict in (e3.predict, module.predict):
                    p = predict(query, labels)
                    self.assertTrue(math.isfinite(p) and .001 <= p <= .999)

    def test_range_for_large_residuals(self):
        for mode in ("probability", "logit"):
            for sign in (-1, 1):
                p = e3.apply_shift(.5, (1000, sign*1000., sign*20000.), .5, 1, mode)
                self.assertTrue(.001 <= p <= .999)
        self.assertEqual(e3.apply_shift(.5, (1000, 1000., 20000.), .5, 1, "probability"), .999)

    def test_stateless_and_no_mutation(self):
        x = [SUBJECT, item()]
        labels = [[[OTHER, item()], 1]]
        before = copy.deepcopy((x, labels))
        p = e3.predict(x, labels)
        e3.predict([OTHER, item("unrelated")], [[[SUBJECT, item()], 0]])
        self.assertEqual(e3.predict(x, labels), p)
        self.assertEqual((x, labels), before)

    def test_embedded_e2_weights_unchanged_and_e3_matches_variant(self):
        module = submission()
        self.assertEqual(module.PARAMETERS, e2.PARAMETERS)
        self.assertEqual(module.GAMMA, e2.GAMMA)
        rng = random.Random(53)
        for _ in range(25):
            labels = [[[dict(OTHER, normalized_name=f"other-{j}"),
                        item() if j % 2 else item("Synthetic distinct")], rng.randrange(2)] for j in range(15)]
            x = [SUBJECT, item()]
            self.assertEqual(module.predict(x, labels), e3.predict(x, labels, **module.E3_CONFIG))
            self.assertEqual(module.predict_e2(x, labels), e2.predict(x, labels))


if __name__ == "__main__":
    unittest.main()
