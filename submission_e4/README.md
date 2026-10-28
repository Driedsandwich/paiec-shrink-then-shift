# Three-level shrinkage predictor (E4)

E4 extends the day-one predictor (E1) with one more pooling level so that the
overall success rate of the evaluated population is learned during evaluation,
from labels revealed within the budget:

1. Global rate: acquired labels on the *other* anonymous benchmarks (all
   subjects), shrunk toward `PRIOR_MEAN = 0.5` with `GLOBAL_STRENGTH`
   pseudo-counts.
2. Benchmark rate: acquired labels of the other subjects on the same
   `benchmark_id`, shrunk toward the global rate with `BENCH_STRENGTH`.
3. Pair rate: labels of the same subject (complete visible attributes) on the
   same benchmark, shrunk toward the benchmark rate with `PAIR_STRENGTH`.

Each acquired label enters exactly one level. At budget 0, or when
`benchmark_id` is missing, the prediction is 0.5, whose Brier score is 0.25 for
any base rate. Without labels on other benchmarks, E4 returns exactly the E1
prediction. Predictions are clipped to [0.001, 0.999]. Only the evidence
supplied to the current call is used; nothing is retained between calls or
budgets. No `labeling.py` is included, so the platform's default random
acquisition applies. No training data, model files, API credentials, or
third-party packages are used.

`GLOBAL_STRENGTH` was selected on the tuning benchmarks of measurement-db with
the local simulator, averaging a standard condition (2025-2026 subjects) and a
low-base-rate condition (all release years, pairs with public success rate
below 0.55). Held-out benchmarks were not used for the selection.

Development used an LLM-based coding assistant; this is disclosed here and in
the technical report.
