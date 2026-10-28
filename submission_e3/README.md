# E3: cross-subject same-item residuals on frozen E2

E2 is a hierarchical E1b probability plus a centered, ridge-fitted item-feature
logit offset. Its parameters and weights are embedded unchanged in model.py.
E1b pools acquired labels for the same anonymous benchmark, using prior mean
0.597568381426247 and benchmark/pair shrinkage strengths of 4. E2 uses gamma=1
and ridge penalty=10. Its 14 bounded string/metadata features describe length,
line count, digits, non-ASCII-letter fraction, compression, code/math/choice
cues, URLs, interactors, feature-key count, year gap, and missing-year flags.

E3 matches all four item fields exactly: item_content, item_features,
interactors, and benchmark_id. Only entries whose subject dictionary differs
from the target subject contribute a residual. No anonymous identity is
reverse engineered. For each match (t,y), q_t is E1b for subject t on this
benchmark, recomputed from the entire supplied labeled list. The matching
label stays in q_t (in-sample residual shrinkage); target-subject labels may
influence q_t's E1b benchmark prior, but never contribute a residual.

Probability mode: clip(p_E2 + eta * sum(y-q_t)/(n+lambda)).
Logit mode: clip(sigmoid(logit(p_E2) + eta * sum((y-q_t)/
max(q_t*(1-q_t),0.05))/(n+lambda))). The variance floor bounds extreme
approximate logit residuals. All matching acquired entries are counted.
Selected configuration: lambda=0.5, eta=0.5, mode=probability.

The grid was lambda in {0.5,1,2,4,8}, eta in {0.5,1}, and both modes;
selection minimized balanced Brier ALC on tuning seeds 0..19. The split
uses SHA256 benchmark hashing with holdout_fraction=0.3. Confirmation
seeds 1000..1049 and packed sensitivity were not used for selection.

Training data: measurement-db revision 54074453 (requester-specified;
no network verification), binary item benchmarks, subjects released in
2025-2026, and eligible subject-benchmark pairs with at least 80 items.
Only the tuning benchmarks below were used for E2 training and E3 tuning.

- algotune
- appworld
- benger
- editbench
- engdesign
- hle
- medhelm_head_qa
- medhelm_med_mcqa
- medhelm_medbullets
- medhelm_medcalc_bench
- medhelm_medhallu
- medhelm_pubmed_qa
- openbiorq
- risebench
- software_world_models
- workbench_revisited

Only the Python standard library is required. No persistent call state, I/O,
or custom acquisition function is used. Outputs are clipped to [0.001,0.999].
No matching other-subject label returns E2 exactly; malformed evidence is
skipped and correction failures return E2. Item text is never logged.

Disclosure: Anthropic Claude and OpenAI Codex provided LLM-based coding
assistance for implementation, experiment orchestration, and documentation.

Limitation: availability of same-item labels in the official evaluation
stream is not confirmed. The local simulator is not an official-equivalence
claim, and sparse overlap can make E3 behave almost exactly like E2.
