# Shrink, Then Adapt — PAIEC 2026 submission code

(The repository name reflects an earlier working title.)

Code for a submission to the NeurIPS 2026 Predictive AI Evaluation Competition (PAIEC):
predicting whether an AI subject answers an item correctly, from at most 31 acquired
labels per subject–benchmark pair, on benchmarks unseen in training (Brier ALC).

## Contents

| Path | What it is |
|---|---|
| `submission_e1/` … `submission_e4/` | Predictors (`model.py`, standard library only, stateless). E1: shrinkage with prior 0.5 and pseudo-counts 4 and 4 (the code comment about replacing these day-one values predates the report; E1 keeps them); E1b: hierarchical shrinkage with the fitted prior; E2: + item-feature shift; E3: + same-item residual shift; E4 (selected): E1 plus a global rate learned during evaluation from labels on the other benchmarks (prior 0.5 at budget 0) |
| `f1/` | Local simulator of the streaming evaluation (`run_simulation`) and the SHA-256 tuning/holdout split |
| `variants/` | V1/V2/V3, constant 0.5, Empirical Mean, E2 features, E3 shift |
| `tools/` | Tuning and comparison scripts (`tune_e1.py`, `train_e2.py`, `train_e3.py`), decomposition (`decompose.py`), ZIP builder (`build_zip.py`), figure and table regeneration from archived results (`make_figures.py`) |
| `benchmarks.txt` | The 66 measurement-db benchmarks whose four tables were downloaded |
| `tools/official/` | Organizers' `check_submission_zip.py`, `submit_api.py` and evaluator client, copied unchanged (MIT, see `tools/official/LICENSE`) |
| `tests/` | Unit tests (synthetic data only) |
| `experiments/` | Result files behind the technical report (local holdout, 50 seeds) |
| `experiments/base_rate/` | Public success rates of eligible pairs (`pair_rates.py`), the standard vs low-base-rate comparison of E1/E1b/E4 (`shift_experiment_cached.py`, with `--verify` to check equality with the full simulator path), and the table writer (`make_results_md.py`) |

`submission_*/README.md` files are kept as packaged in the submission ZIPs (`tools/build_zip.py`
includes them), so some of their wording predates this README.

## Data

Training and local evaluation use only measurement-db
(<https://huggingface.co/datasets/aims-foundations/measurement-db>, revision
`54074453da7b9911be0f48e4c560be547bc84d21`; gated access, so accept its terms first;
CC BY-SA 4.0 for the curation, upstream licenses apply to individual benchmarks).
No raw data (parquet tables, item text or individual responses) is included in this
repository. The JSON and Markdown result files under `experiments/` and `benchmarks.txt`
contain statistics and identifiers (benchmark, subject and item IDs) derived from
measurement-db, for example per subject–benchmark public success rates in
`experiments/base_rate/pair_rates.json`. They are shared under CC BY-SA 4.0 with
attribution to measurement-db, subject to the upstream licenses of the individual
benchmarks. For each benchmark in `benchmarks.txt`, fetch
`<benchmark>/formatted_tables/{benchmarks,subjects,items,responses}.parquet` at that
revision (264 files, 123 MB); traces and assets are not needed. Set `DATA` to the folder
that contains the benchmark subdirectories.

## Reproduce

Python 3.12. `pip install -r requirements-dev.txt` installs pyarrow, numpy, matplotlib
and requests (requests only for the organizers' submission client); the predictors use the
Python standard library only. The data commands overwrite the archived results in
`experiments/` and the constants embedded in `submission_e2/model.py` and
`submission_e3/model.py`; `git diff --stat` afterwards shows any difference.
`tools/train_e2.py` and `tools/train_e3.py` also regenerate `submission_e2/README.md` and
`submission_e3/README.md` from templates in the scripts, which may differ from the submitted
files (the E2 template is in Japanese); restore them with
`git checkout -- submission_e2/README.md submission_e3/README.md`. Most result files also
record timing and memory fields (such as `seconds`, `load_seconds` and `max_rss_mib`), so
compare the model constants and ALC values rather than whole files. Every script above that
splits benchmarks uses a holdout fraction of 0.3; `f1/simulate.py` defaults to 0.2, so pass
`--holdout-fraction 0.3` when running F1 directly.

```
pip install -r requirements-dev.txt
DATA=/absolute/path/to/measurement-db-snapshot

# Earlier methods (E1-E3) and the error decomposition
PYTHONPATH=. python -B tools/tune_e1.py --data "$DATA"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. python -B tools/train_e2.py --data "$DATA"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. python -B tools/train_e3.py --data "$DATA"
PYTHONPATH=. python -B tools/decompose.py --data "$DATA" \
  --out experiments/decomposition/holdout_50seeds.json --runs 50 --seed 1000

# Base-rate experiments and E4 (GLOBAL_STRENGTH = argmin of the mean tuning ALC = 8)
python -B experiments/base_rate/pair_rates.py "$DATA" experiments/base_rate/pair_rates.json
for cond in standard low; do
  python -B experiments/base_rate/shift_experiment_cached.py --data "$DATA" \
    --rates experiments/base_rate/pair_rates.json --split tune --condition $cond \
    --runs 30 --seed 0 --verify 2 --out experiments/base_rate/tune_$cond.json
done
for cond in standard low; do
  python -B experiments/base_rate/shift_experiment_cached.py --data "$DATA" \
    --rates experiments/base_rate/pair_rates.json --split holdout --condition $cond \
    --strengths 8 --runs 50 --seed 1000 --verify 2 --out experiments/base_rate/holdout_$cond.json
done
python -B experiments/base_rate/make_results_md.py E4_RESULTS.md

# Tests, packaging, figures (no data needed)
PYTHONPATH=. python -B -m unittest discover -s tests -p "test_*.py"
PYTHONPATH=. python -B tools/build_zip.py submission_e4
python -I -B tools/official/check_submission_zip.py dist/submission_e4.zip
python -B tools/make_figures.py --out regenerated
```

`tune_e1.py` has a 20-minute guard; any skipped configurations are listed in
`grid_reductions` of `experiments/e1_tuning/results.json` (none in our run). If Matplotlib
cannot build its font cache, point `MPLCONFIGDIR` to a writable folder.

## Local results (7 holdout benchmarks, 50 seeds; not official scores)

Earlier methods (full-text runs):

| Method | Brier ALC (mean ± SD) |
|---|---:|
| Constant 0.5 | 0.25000 |
| Empirical Mean (organizer baseline) | 0.23206 ± 0.02723 |
| E1b | 0.19412 ± 0.01735 |
| E2 | 0.19394 ± 0.01732 |
| E3 | 0.19359 ± 0.01708 |

These values come from `experiments/decomposition/holdout_50seeds.json` (constant,
Empirical Mean, E1b, E2) and `experiments/e3_item/results.json` (E3).
`experiments/e1_tuning/` records the earlier tuning run on the text-blanked harness (item
text not read), so its holdout values differ (for example, E1b 0.19360 and Empirical Mean
0.23084).

Standard (2025–2026 subjects) versus low-base-rate (all release years, pairs with public
success rate below 0.55) holdout conditions, 50 paired seeds:

| Method | Standard ALC | Low ALC |
|---|---:|---:|
| E1 | 0.19803 | 0.22430 |
| E1b | 0.19412 | 0.23223 |
| E4 (c = 8) | 0.19770 | 0.22388 |

See the technical report for the protocol, the error decomposition, and limitations.

## Disclosure

LLM-based coding assistants (Anthropic Claude and OpenAI Codex) were used for coding,
experiment orchestration, and drafting.

## License

Code and documentation: MIT (see `LICENSE`), unless stated otherwise. Files under
`tools/official/` keep their original MIT license. The JSON and Markdown result files under
`experiments/` and `benchmarks.txt` are CC BY-SA 4.0, subject to the upstream licenses of
the individual benchmarks (see Data).
