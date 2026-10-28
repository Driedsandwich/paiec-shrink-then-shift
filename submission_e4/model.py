"""Three-level shrinkage of acquired labels (Python standard library only).

Budget 0 returns PRIOR_MEAN = 0.5, the constant whose Brier score is 0.25 for
every base rate. With labels, a global rate is learned from the acquired labels
on the *other* anonymous benchmarks and shrunk toward PRIOR_MEAN; the benchmark
rate pools the other subjects' labels on the same benchmark and is shrunk toward
the global rate; the subject-benchmark rate is shrunk toward the benchmark rate.
Each label enters exactly one level. Everything is recomputed from the supplied
``labeled`` evidence on each call, so no state is kept between calls, workers,
or budgets.
"""

import math

PRIOR_MEAN = 0.5
GLOBAL_STRENGTH = 8.0  # selected on tuning benchmarks (standard + low-base-rate conditions)
BENCH_STRENGTH = 4.0  # pseudo-counts pulling the benchmark rate toward the global rate
PAIR_STRENGTH = 4.0  # pseudo-counts pulling the pair rate toward the benchmark rate
EPS = 1e-3


def _clip(p: float) -> float:
    if not math.isfinite(p):
        return PRIOR_MEAN
    return min(1.0 - EPS, max(EPS, p))


def predict_with_strength(input, labeled, global_strength):
    try:
        subject, item = input
    except (TypeError, ValueError):
        return _clip(PRIOR_MEAN)
    benchmark = item.get("benchmark_id") if isinstance(item, dict) else None
    if not labeled or not isinstance(benchmark, str) or not benchmark:
        return _clip(PRIOR_MEAN)

    global_s = global_n = other_s = other_n = pair_s = pair_n = 0
    for entry in labeled:
        try:
            (observed_subject, observed_item), label = entry
        except (TypeError, ValueError):
            continue
        if not isinstance(observed_item, dict) or label not in (0, 1):
            continue
        observed_benchmark = observed_item.get("benchmark_id")
        if not isinstance(observed_benchmark, str) or not observed_benchmark:
            continue
        y = int(label)
        if observed_benchmark != benchmark:
            global_s += y
            global_n += 1
        elif observed_subject == subject:
            pair_s += y
            pair_n += 1
        else:
            other_s += y
            other_n += 1

    global_rate = (global_s + global_strength * PRIOR_MEAN) / (global_n + global_strength)
    bench_rate = (other_s + BENCH_STRENGTH * global_rate) / (other_n + BENCH_STRENGTH)
    p = (pair_s + PAIR_STRENGTH * bench_rate) / (pair_n + PAIR_STRENGTH)
    return _clip(float(p))


def predict(input: list, labeled: list | None = None) -> float:
    return predict_with_strength(input, labeled, GLOBAL_STRENGTH)
