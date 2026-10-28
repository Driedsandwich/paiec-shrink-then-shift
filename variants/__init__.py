"""Stateless E1 predictors. No imports outside the Python standard library."""

import math

EPS = 0.001


def clip(p):
    return max(EPS, min(1 - EPS, p)) if math.isfinite(p) else 0.5


def logit(p):
    p = clip(p)
    return math.log(p / (1 - p))


def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    z = math.exp(x)
    return z / (1 + z)


def _number(value, default, positive=False):
    try:
        value = float(value)
        return value if math.isfinite(value) and (not positive or value > 0) else default
    except Exception:
        return default


def _key(value):
    # Subject dictionaries have the same identity semantics as E1 equality.
    if isinstance(value, dict):
        return tuple(sorted((str(k), _key(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_key(v) for v in value)
    return value


def identify(input):
    subject, item = input
    benchmark = item.get("benchmark_id") if isinstance(item, dict) else None
    if not isinstance(benchmark, str) or not benchmark:
        raise ValueError("missing benchmark")
    subject = _key(subject)
    hash(subject)
    return subject, benchmark


def summarize(labeled):
    """Counts of valid acquired entries only, keyed by (subject, benchmark)."""
    counts = {}
    if not isinstance(labeled, (list, tuple)):
        return counts
    for entry in labeled:
        try:
            input, y = entry
            if y not in (0, 1):
                continue
            key = identify(input)
            s, n = counts.get(key, (0, 0))
            counts[key] = (s + int(y), n + 1)
        except Exception:
            continue
    return counts


def from_counts(subject, benchmark, counts, *, variant="V2", k=4, b=4, a=8,
                prior_mean=0.5):
    """Shared scalar formulas used by both predict and the experiment replay.

    V3 uses clipped empirical rates on other benchmarks, weights by the own
    subject's label count, and skips benchmarks with no other-subject labels.
    Its offset is sum(n * log_odds_difference) / (sum(n) + a).
    Empirical Mean deliberately retains exact 0 and 1 (no smoothing/clipping).
    """
    m = clip(_number(prior_mean, 0.5))
    try:
        if variant == "constant":
            return 0.5
        s, n = counts.get((subject, benchmark), (0, 0))
        if variant == "empirical":
            return s / n if n else 0.5
        k = _number(k, 4, True)
        b = _number(b, 4, True)
        a = _number(a, 8, True)
        if variant not in ("V1", "V2", "V3"):
            return m
        prior = m
        if variant != "V1":
            os = on = 0
            for (sid, bid), (ss, nn) in counts.items():
                if bid == benchmark and sid != subject:
                    os += ss
                    on += nn
            prior = (os + b * m) / (on + b)
        if variant == "V3":
            weighted = total = 0
            for (sid, bid), (ss, nn) in counts.items():
                if sid != subject or bid == benchmark or not nn:
                    continue
                os = on = 0
                for (other, other_bid), (other_s, other_n) in counts.items():
                    if other_bid == bid and other != subject:
                        os += other_s
                        on += other_n
                if on:
                    weighted += nn * (logit(ss / nn) - logit(os / on))
                    total += nn
            prior = sigmoid(logit(prior) + weighted / (total + a))
        return clip((s + k * prior) / (n + k))
    except Exception:
        return m


def predict(input, labeled=None, *, variant="V2", k=4, b=4, a=8, prior_mean=0.5):
    """Never raises for malformed ordinary Python inputs; retains no state."""
    try:
        subject, benchmark = identify(input)
        return from_counts(subject, benchmark, summarize(labeled), variant=variant,
                           k=k, b=b, a=a, prior_mean=prior_mean)
    except Exception:
        return 0.5 if variant in ("constant", "empirical") else clip(_number(prior_mean, 0.5))


def v1(input, labeled=None, *, k=4, prior_mean=0.5):
    return predict(input, labeled, variant="V1", k=k, prior_mean=prior_mean)


def v2(input, labeled=None, *, k=4, b=4, prior_mean=0.5):
    return predict(input, labeled, variant="V2", k=k, b=b, prior_mean=prior_mean)


def v3(input, labeled=None, *, k=4, b=4, a=8, prior_mean=0.5):
    return predict(input, labeled, variant="V3", k=k, b=b, a=a, prior_mean=prior_mean)


def constant(input, labeled=None):
    return 0.5


def empirical(input, labeled=None):
    return predict(input, labeled, variant="empirical")
