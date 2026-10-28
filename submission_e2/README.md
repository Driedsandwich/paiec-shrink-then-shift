# E2: hierarchical shrinkage plus an item-level shift

E2 adds a relative item ease score to the E1b estimate on the logit scale:

    p = sigmoid(logit(p_E1b) + gamma * (f(x) - c))

- `p_E1b` is the hierarchical shrinkage estimate (pair labels shrunk toward the
  other subjects' rate on the same anonymous `benchmark_id`, shrunk toward an
  equal-weight prior of 0.5976; strengths k = b = 4).
- `f(x)` is a ridge regression on item-string features: log length, log line
  count, digit share, non-ASCII-letter share, zlib compression ratio, code /
  math / multiple-choice cues, URL presence, interactors presence, number of
  top-level `item_features` keys, and the gap between the latest year mentioned
  in the item and the subject's release year (with a missing-year indicator).
  Feature scanning uses the first and last 4,096 characters; length uses the
  full string. Benchmark identifiers and model names are not features.
- `c` is the mean of `f` over acquired items of the same benchmark in
  `labeled`, or the training mean when there are none.
- gamma = 1.0 and ridge penalty = 10.0. Weights, means, and scales are embedded
  in `model.py`.

The predictor uses the Python standard library only, keeps no state between
calls, clips outputs to [0.001, 0.999], and falls back to the E1b value on any
error. No `labeling.py` is included (default random acquisition).

Training data: measurement-db, revision 54074453da7b9911be0f48e4c560be547bc84d21
(binary, item-level benchmarks; subjects released in 2025–2026; pairs with at
least 80 items). Only the tuning benchmarks were used: algotune, appworld,
benger, editbench, engdesign, hle, medhelm_head_qa, medhelm_med_mcqa,
medhelm_medbullets, medhelm_medcalc_bench, medhelm_medhallu, medhelm_pubmed_qa,
openbiorq, risebench, software_world_models, workbench_revisited. The ridge
penalty was chosen by leave-one-benchmark-out cross-validation within these
benchmarks and gamma by simulated Brier ALC on them; holdout benchmarks were
not used for any choice.

Disclosure: LLM-based coding assistants (Anthropic Claude and OpenAI Codex)
were used for coding, experiment orchestration, and drafting.
