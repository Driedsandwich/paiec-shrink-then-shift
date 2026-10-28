# Hierarchical shrinkage predictor (E1b, tuned prior)

For each target, the predictor pools acquired labels at two levels:

1. Benchmark rate: acquired labels of the other subjects on the same anonymous
   `benchmark_id`, shrunk toward `PRIOR_MEAN` with `BENCH_STRENGTH`
   pseudo-counts. The target pair's own labels are not included here, so they
   are counted only once.
2. Pair rate: labels of the same subject (complete visible attributes) on the
   same benchmark, shrunk toward the benchmark rate with `PAIR_STRENGTH`
   pseudo-counts.

At budget 0, or when `benchmark_id` is missing, the prediction is `PRIOR_MEAN`.
Predictions are clipped to [0.001, 0.999]. Only the evidence supplied to the
current call is used; nothing is retained between calls or budgets. No
`labeling.py` is included, so the platform's default random acquisition applies.
No training data, model files, API credentials, or third-party packages are used.

`PRIOR_MEAN` (0.5976) is the equal-weight mean of subject-benchmark success
rates on the tuning benchmarks of measurement-db (revision 54074453). The
strengths `BENCH_STRENGTH = PAIR_STRENGTH = 4` were selected on the same tuning
benchmarks with a local simulator of the streaming evaluation.

Development used an LLM-based coding assistant; this is disclosed here and will
be disclosed in the technical report.
