"""Hierarchical shrinkage of acquired labels (Python standard library only).

Budget 0 returns PRIOR_MEAN. With labels, the benchmark rate pools the other
subjects' acquired labels on the same anonymous ``benchmark_id`` and is shrunk
toward PRIOR_MEAN; the subject-benchmark rate is shrunk toward that benchmark
rate. The target pair's own labels enter only once (in the pair rate).
Everything is recomputed from the supplied ``labeled`` evidence on each call,
so no state is kept between calls, workers, or budgets.
"""

import math

# PRIOR_MEAN is the equal-weight mean of subject-benchmark success rates on the
# tuning benchmarks of measurement-db (revision 54074453, 2025-2026 subjects,
# pairs with at least 80 items). The strengths were selected on the same tuning
# benchmarks with the local evaluation simulator (k = b = 4 was best).
PRIOR_MEAN = 0.597568381426247
BENCH_STRENGTH = 4.0  # pseudo-counts pulling the benchmark rate toward PRIOR_MEAN
PAIR_STRENGTH = 4.0  # pseudo-counts pulling the pair rate toward the benchmark rate
EPS = 1e-3


def _clip(p: float) -> float:
    if not math.isfinite(p):
        return PRIOR_MEAN
    return min(1.0 - EPS, max(EPS, p))


def predict(input: list, labeled: list | None = None) -> float:
    subject, item = input
    benchmark = item.get("benchmark_id") if isinstance(item, dict) else None
    if not labeled or not isinstance(benchmark, str) or not benchmark:
        return _clip(PRIOR_MEAN)

    other_s = other_n = pair_s = pair_n = 0
    for entry in labeled:
        try:
            (observed_subject, observed_item), label = entry
        except (TypeError, ValueError):
            continue
        if not isinstance(observed_item, dict) or observed_item.get("benchmark_id") != benchmark:
            continue
        if label not in (0, 1):
            continue
        y = int(label)
        if observed_subject == subject:
            pair_s += y
            pair_n += 1
        else:
            other_s += y
            other_n += 1

    bench_rate = (other_s + BENCH_STRENGTH * PRIOR_MEAN) / (other_n + BENCH_STRENGTH)
    p = (pair_s + PAIR_STRENGTH * bench_rate) / (pair_n + PAIR_STRENGTH)
    return _clip(float(p))
