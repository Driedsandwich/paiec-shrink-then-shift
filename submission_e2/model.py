"""Bounded, text-free numeric features and stateless E2 inference (stdlib only).

Input text is inspected, never returned or logged. No benchmark/subject identity
is a regressor. Feature scans use at most 4096 characters (head and tail);
the character count uses the full input. The same rule applies in training.
"""

import json
import math
import re
import zlib

PRIOR_MEAN = 0.597568381426247
SCAN_CHARS = 4096
FEATURE_NAMES = (
    "log_chars", "log_estimated_lines", "digit_fraction", "non_ascii_letter_fraction",
    "compression_ratio", "log_code_markers", "log_math_markers", "log_choice_markers",
    "has_url", "has_interactors", "feature_key_count", "year_minus_release",
    "no_year", "no_release_year",
)
_DELETE_DIGITS = str.maketrans("", "", "0123456789")
_DELETE_LETTERS = str.maketrans("", "", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
_CODE_WORDS = re.compile(r"\b(?:def|class|import)\b")
_MATH = re.compile(r"\\(?:frac|sum|int|sqrt|begin|alpha|beta|theta)\b")
_CHOICE = re.compile(r"[A-H][).]|\([A-H]\)")
_URL = re.compile(r"https?://|www\.", re.IGNORECASE)
_YEAR = re.compile(r"(?<!\d)(?:1\d{3}|20\d{2}|21\d{2})(?!\d)")


def _text(value):
    return value if isinstance(value, str) else ""


def extract_features(item, subject=None):
    """Return a fixed dictionary of finite numbers for ordinary malformed inputs.

    Years are bounded to 1000..2199; the signed year gap is clipped to +/-100.
    Line count is estimated from the bounded scan. Compression uses zlib level 1.
    Empty JSON interactors are absent. Only top-level item_features keys count.
    """
    values = dict.fromkeys(FEATURE_NAMES, 0.0)
    values.update(no_year=1.0, no_release_year=1.0)
    try:
        item = item if isinstance(item, dict) else {}
        subject = subject if isinstance(subject, dict) else {}
        full = _text(item.get("item_content"))
        n = len(full)
        text = full if n <= SCAN_CHARS else full[:SCAN_CHARS // 2] + full[-SCAN_CHARS // 2:]
        size = len(text)
        values["log_chars"] = math.log1p(n)
        lines = 0 if not size else 1 + text.count("\n") * n / size
        values["log_estimated_lines"] = math.log1p(lines)
        if size:
            values["digit_fraction"] = (size - len(text.translate(_DELETE_DIGITS))) / size
            letters = size - len(text.translate(_DELETE_LETTERS))
            values["non_ascii_letter_fraction"] = 1 - letters / size
            encoded = text.encode("utf-8", errors="replace")
            values["compression_ratio"] = len(zlib.compress(encoded, 1)) / len(encoded)
        code = text.count("```") + sum(text.count(c) for c in "{};") + len(_CODE_WORDS.findall(text))
        values["log_code_markers"] = math.log1p(code)
        math_count = len(_MATH.findall(text)) if "\\" in text else 0
        values["log_math_markers"] = math.log1p(text.count("$") + math_count)
        choices = 0
        if any(c in text for c in "ABCDEFGH") and (")" in text or "." in text):
            choices = sum(1 for m in _CHOICE.finditer(text)
                          if m.start() == 0 or text[m.start() - 1].isspace() or text[m.start() - 1] == ":")
        values["log_choice_markers"] = math.log1p(choices)
        folded = text.casefold()
        values["has_url"] = float(bool(_URL.search(text))) if ("http" in folded or "www." in folded) else 0.0
        interactors = item.get("interactors")
        if isinstance(interactors, str):
            interactors = interactors.strip()
            if interactors in ("", "null", "[]", "{}", '""'):
                interactors = None
        values["has_interactors"] = float(bool(interactors))
        extra = item.get("item_features")
        if isinstance(extra, str):
            try:
                extra = json.loads(extra) if len(extra) <= 65536 else None
            except Exception:
                extra = None
        values["feature_key_count"] = float(len(extra)) if isinstance(extra, dict) else 0.0
        years = [int(m.group()) for m in _YEAR.finditer(text)]
        release = _YEAR.search(_text(subject.get("release_date"))[:32])
        values["no_year"] = float(not years)
        values["no_release_year"] = float(release is None)
        if years and release:
            values["year_minus_release"] = float(max(-100, min(100, max(years) - int(release.group()))))
    except Exception:
        # Never expose content through exception strings, even on invalid metadata.
        pass
    return {name: float(v) if isinstance(v, (int, float)) and math.isfinite(v) else 0.0
            for name, v in values.items()}


def feature_vector(item, subject=None):
    values = extract_features(item, subject)
    return [values[name] for name in FEATURE_NAMES]


def clip(p):
    return max(0.001, min(0.999, p)) if math.isfinite(p) else PRIOR_MEAN


def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


def e1b(input, labeled=None):
    """E1b formula; malformed evidence is skipped, malformed input uses the prior."""
    try:
        subject, item = input
        benchmark = item.get("benchmark_id") if isinstance(item, dict) else None
        if not isinstance(benchmark, str) or not benchmark or not isinstance(labeled, (list, tuple)):
            return PRIOR_MEAN
        own_s = own_n = other_s = other_n = 0
        for entry in labeled:
            try:
                (observed_subject, observed_item), y = entry
                if y not in (0, 1) or not isinstance(observed_item, dict):
                    continue
                if observed_item.get("benchmark_id") != benchmark:
                    continue
                if observed_subject == subject:
                    own_s += int(y)
                    own_n += 1
                else:
                    other_s += int(y)
                    other_n += 1
            except Exception:
                continue
        prior = (other_s + 4 * PRIOR_MEAN) / (other_n + 4)
        return clip((own_s + 4 * prior) / (own_n + 4))
    except Exception:
        return PRIOR_MEAN


def feature_score(item, subject, parameters):
    x = feature_vector(item, subject)
    score = math.fsum(w * ((v - m) / s) for v, w, m, s in zip(
        x, parameters["weights"], parameters["means"], parameters["scales"]))
    score += parameters.get("intercept", 0.0)
    if not math.isfinite(score):
        raise ValueError("Non-finite feature score")
    return score


def predict_with_parameters(input, labeled, parameters, gamma):
    """E4: sigmoid(logit(E1b) + gamma*(f(x)-same-benchmark labeled mean))."""
    baseline = e1b(input, labeled)
    try:
        if gamma == 0:
            return baseline  # exact equality, without a logit/sigmoid round trip
        subject, item = input
        benchmark = item.get("benchmark_id")
        if not isinstance(benchmark, str) or not benchmark:
            return baseline
        # This cache exists only during this call. Its key covers exactly the
        # text/metadata read by extract_features, including original text length.
        # Repeated items across subjects/labels need not be compressed again.
        cache = {}

        def score_once(observed_item, observed_subject):
            key = None
            if isinstance(observed_item, dict) and isinstance(observed_subject, dict):
                full = _text(observed_item.get("item_content"))
                extra = observed_item.get("item_features")
                interactors = observed_item.get("interactors")
                if isinstance(extra, str) and isinstance(interactors, str):
                    sample = full if len(full) <= SCAN_CHARS else full[:SCAN_CHARS // 2] + full[-SCAN_CHARS // 2:]
                    key = (len(full), sample, extra if len(extra) <= 65536 else None,
                           bool(interactors.strip() not in ("", "null", "[]", "{}", '""')),
                           _text(observed_subject.get("release_date"))[:32])
            if key is None:
                return feature_score(observed_item, observed_subject, parameters)
            if key not in cache:
                cache[key] = feature_score(observed_item, observed_subject, parameters)
            return cache[key]

        scores = []
        for entry in labeled or ():
            try:
                (observed_subject, observed_item), y = entry
                if (y in (0, 1) and isinstance(observed_item, dict) and
                        observed_item.get("benchmark_id") == benchmark):
                    scores.append(score_once(observed_item, observed_subject))
            except Exception:
                continue
        center = math.fsum(scores) / len(scores) if scores else parameters["global_mean"]
        offset = gamma * (score_once(item, subject) - center)
        p = sigmoid(math.log(baseline / (1 - baseline)) + offset)
        return clip(p)
    except Exception:
        return baseline


PARAMETERS = {'feature_names': ['log_chars', 'log_estimated_lines', 'digit_fraction', 'non_ascii_letter_fraction', 'compression_ratio', 'log_code_markers', 'log_math_markers', 'log_choice_markers', 'has_url', 'has_interactors', 'feature_key_count', 'year_minus_release', 'no_year', 'no_release_year'], 'means': [6.997887917790627, 2.2890122642477206, 0.015767048150030764, 0.25413010982287004, 0.6093801941175365, 0.982853872190844, 0.1251954198915513, 0.4688989240903564, 0.026431101565516577, 0.0, 0.0, -4.264377546882405, 0.8666251863713764, 0.0], 'scales': [1.879994624279932, 1.9765894090108156, 0.031079445330314627, 0.09054754516621964, 0.18834427889498495, 1.272137378272917, 0.5734833985588108, 0.7695599390775066, 0.16041352323152214, 1.0, 1.0, 19.254884352573637, 0.33997937101870446, 1.0], 'weights': [-0.002716435085710775, -0.00045408112652707544, -0.005589875506984316, -0.0025753215510360395, 0.0033953447079799735, -0.0024039229986865893, 0.0017356385075921262, -0.00013308751211021873, 0.0013753107962375655, 0.0, 0.0, 0.0012514736465196262, 0.0001719060618327828, 0.0], 'intercept': -1.9949319973733282e-17, 'global_mean': -1.5851035761738075e-16, 'alpha': 10.0, 'scan_chars': 4096}
GAMMA = 1.0

def predict(input, labeled=None):
    return predict_with_parameters(input, labeled, PARAMETERS, GAMMA)
