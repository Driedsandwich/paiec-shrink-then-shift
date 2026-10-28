# F1: local simulator of the PAIEC streaming evaluation

## What F1 is

F1 approximates the streaming Brier ALC of PAIEC (NeurIPS 2026 Predictive AI Evaluation Competition) on public data available locally. We do not claim that it is identical to the official scorer. Brier is the squared error between a predicted probability and the actual 0/1 response. ALC is a fixed weighted average of the Brier scores as the number of acquired labels grows. Lower is better.

## How to run

Run from the repository root. F1 needs pyarrow (tested with Python 3.12). It uses only the Python standard library and pyarrow, and it makes no network calls and downloads no data.

```sh
python -B f1/simulate.py --data /path/to/data --model submission_e1/model.py \
  --benchmarks benchmark_a,benchmark_b --runs 10 --seed 42 \
  --pairs-strategy balanced --split tune --holdout-fraction 0.3 \
  --out results.json
```

`--data` is a directory with the structure below. You can also point it at a single dataset directory instead of its parent.

```text
data/
  dataset_a/formatted_tables/
    benchmarks.parquet
    subjects.parquet
    items.parquet
    responses.parquet
  dataset_b/formatted_tables/...
```

With `--labeling /path/to/labeling.py`, F1 uses the `acquisition_function` in that file. Without it, F1 draws candidates with the same hash-based rule as the organizers' evaluator. The predictor must expose `predict(input, labeled)`.

| Argument | Role and default |
|---|---|
| `--benchmarks a,b` | Restricts the run to these original benchmark_ids. If omitted, all candidates are used. An ID that is not found is an error. |
| `--runs N` / `--seed S` | Number of repetitions; the default is 1 run (default seed 0). The seed of each run is `S + 0, S + 1, ...`. |
| `--pairs-strategy balanced` | Shuffles the pairs with a seed and selects up to (cap ÷ 80, rounded down) pairs. After reserving 80 items for each selected pair, the remainder is distributed one item at a time in round-robin order. |
| `--pairs-strategy packed` | Shuffles the pairs the same way and gives each pair, in order, as many items as it has. Stops when fewer than 80 remain. |
| `--max-sample 1000` | Upper bound on the number of distinct subject–item cells summed over all pairs. It can be reduced within 80–1,000. It is not a count of repeated-trial rows. |
| `--split tune\|holdout\|all` | Selects the tuning split, the holdout split, or both. The default is `all`. Pass `tune` explicitly when choosing constants. |
| `--holdout-fraction F` | Fraction of the hash range assigned to the holdout split (default 0.2; all experiments in this repository use 0.3). It does not guarantee the actual ratio of benchmark counts. |
| `--release-years 2025,2026` | Restricts subjects by release year. By default, subjects with a missing year or a year outside 2025–2026 are excluded. `all` removes the year restriction. |
| `--out results.json` | Output path. The parent directory must already exist. An existing file is replaced. |

The tuning/holdout assignment is decided by `benchmark_split` in [splits.py](splits.py) from all 256 bits of the SHA-256 of the original benchmark_id. A benchmark goes to holdout if the hash divided by `2**256` is below the fraction. The assignment does not depend on the run seed or on the predictor. Use the holdout split last, after constants and features have been chosen. F1 does not automatically detect benchmarks on which a predictor was trained beforehand.

## What was checked against the competition materials

Here, "official" means a copy of the competition page <https://aims.stanford.edu/competition> taken on 2026-10-17 (not redistributed). We did not re-check the live page.

| What was checked | Source |
|---|---|
| The input has 8 subject keys and 4 item keys. All input values are strings, and missing values are empty strings. | Competition page (copy of 2026-10-17), Submission / input field reference |
| Pairs with at least 80 distinct items are eligible, and items are split 50/50. All trials of the same item stay on the same side. | Same page, Test-time Adaptation |
| An acquisition reveals exactly one recorded response of an item. Candidates are offered in a fixed random order, and a rejected candidate is not offered again. | Same page, Acquisition function |
| Acquired labels are shared across all pairs. The context is `subject_id`, `benchmark_id`, `labels_acquired`, `labels_remaining`, `max_labels`, `items_remaining`. The remaining-label count is based on 31. | Same page, Acquisition function |
| All budgets 0/1/3/7/15/31 are evaluated along one trajectory, and every response row is scored. Predictions for the same input are reused. | Same page, Test-time Adaptation / Evaluation |
| Brier is averaged over the responses within a pair, then ALC is computed, and finally pairs are averaged with equal weight. Acquisition and evaluation together use at most 1,000 subject–item cells. | Same page, Computing Brier ALC / Data |
| The default sampling rule, how function arguments are chosen, deep copies, and strict bool return values. Predictions are computed only for the argument forms that are needed, and a TypeError raised inside the function is not retried. | [Organizers' evaluator](../tools/official/tools/streaming_ingestion.py), AcquisitionHook / acquire branch |
| Names and types of the tables and columns that are read. The item's `content` maps to the input's `item_content`, and `interactors` is taken from responses. | The organizers' parquet schema (not redistributed) |

In addition, F1 implements the selection of benchmarks with `response_type` binary and `granularity` item, the release-year filter, anonymized IDs, and the error decomposition.

## Approximations F1 makes on its own

- **Order of eligibility and the cap**: F1 first checks whether a pair has at least 80 distinct items after dropping missing responses. It then selects pairs and caps the total at 1,000. Each selected pair keeps at least 80 items. Pairs with fewer than 80 items are not evaluated.
- **Pair sampling and allocation**: the two strategies `balanced` / `packed` above. Some capacity may remain unused. Within each selected pair, items are drawn uniformly with a seed.
- **Odd counts in the 50/50 split**: if the count is odd, the acquisition side is rounded down and the evaluation side is rounded up. Sampling, splitting, candidate order, and the choice of acquired trial each use an independent random stream.
- **Order across pairs**: lexicographic order of the original `(benchmark_id, subject_id)`. All pairs are advanced to each checkpoint and then evaluated together. An acquisition function called along the way can see the labels of pairs that have already advanced.
- **Anonymization**: the full SHA-256 of the original ID, prefixed with `subject_` / `benchmark_`. The organizers' anonymous IDs themselves are not reproduced. Item IDs are not passed to the predictor. The diagnostic item IDs and the benchmark list in the result JSON keep the original IDs.
- **Choice among repeated trials**: for each acquisition item, one valid recorded response row is drawn uniformly. Rows are sorted by trial, interactors, and label before the seeded draw, so the choice is stable under reordering of the parquet rows.
- **Missing values**: rows whose response is null are excluded from eligibility, acquisition, and scoring. Values other than 0/1 and non-finite values are errors. Non-string input attributes are converted to JSON strings.
- **When candidates run out**: if the acquisition function keeps rejecting, F1 does not force a label purchase. It records the actual number acquired and evaluates later nominal budgets with the same label set. Check `budget_reached: false`.
- **Unit of the decomposition**: normally the item. If trials of the same item have different interactors, cells are split by input. The unit of splitting and acquisition remains the item. This avoids averaging predictions over different inputs inside the formula and keeps agreement with the observed Brier.
- **Aggregation over runs**: the per-run pair averages are averaged again, and the population standard deviation across runs (`ddof=0`) is reported. With one run the standard deviation is 0.
- **Execution environment**: the predictor and the acquisition function run in the same Python process. The organizers' worker respawning, time limits, parallelism, and isolation are not reproduced. Inputs and labels are protected by a deep copy on each call.

These conditions and the reasons for exclusion are recorded in `conditions` and `inventory` of the result JSON. Compare `balanced` and `packed` to see whether differences between methods hold under both. With the same arguments and seed, the sample, the split, and the default acquisition are identical. If an acquisition function itself uses randomness or mutable state, reproducibility also depends on its implementation.

## Outputs and error decomposition

`summary` holds the mean and standard deviation of ALC over all runs, and the mean and standard deviation of Brier per budget. `runs[].pairs[]` records the ALC per pair, the sample counts, and the acquisition and evaluation item IDs. Each `checkpoints` entry records the Brier, the number of response rows scored, the acquired item IDs, the actual number acquired, and the number of labels acquired across all pairs. The model receives only the acquired `[input, label]` pairs, never the result JSON.

The ALC weights are `0.1 B0 + 0.2 B1 + 0.2 B3 + 0.2 B7 + 0.2 B15 + 0.1 B31`. Evaluation predictions are matched by input across all pairs and computed once per checkpoint. They are recomputed at the next checkpoint, where more labels are available. `evaluation_prediction_calls_by_budget` gives the actual number of calls.

For each item (or each item–input cell when needed) with m responses, `π̂` is the sample mean of the responses and `w = m / total number of responses`. `p̄` and `π̄` are the w-weighted means.

```text
level_shift           = (p̄ − π̄)²
shape_mismatch        = Σ w ((p − p̄) − (π̂ − π̄))²
empirical_fluctuation = Σ w π̂(1 − π̂)
Brier                 = sum of the three components above
```

The result is checked against the Brier computed from the actual response rows, and the difference is recorded in `decomposition_residual`. Allowing for floating-point rounding, F1 stops if the absolute difference exceeds `1e-12`. The fluctuation term in this decomposition is a sample quantity, not an unbiased estimate of the true fluctuation. With m=1 the fluctuation of that item is 0. The `m/(m−1)` correction, calibration diagnostics, and AUC are not implemented.

## Memory and limitations

Parquet files are read 8,192 rows at a time, only for the needed columns. traces, assets, and grading_criterion (which contains answer descriptions) are never opened. Eligibility is first checked per dataset, and only the list of pairs and counts is kept. Each run also processes datasets one at a time, and only the inputs and responses of the selected (at most 1,000) subject–item cells are kept across all datasets.

While eligibility is checked, F1 holds the item IDs of that dataset and the set of distinct item IDs per pair in memory. Memory use has limits when a single dataset is very large or when the selected items have many trials or long texts. F1 saves memory by re-reading, so more runs mean more I/O.

The private official data, sample selection, release years, and the other curation criteria as a whole are not reproduced. Success on synthetic tests says nothing about performance on unseen benchmarks or about official rankings. Bias from restricting to small benchmarks also remains. If a duplicate benchmark_id is found in more than one dataset, F1 raises an error instead of counting pairs twice.

External model.py / labeling.py files are executed as ordinary Python code. F1 is not a sandbox that blocks network access, so pass only trusted local code. F1 itself has no network code.

## Tests

Temporary test data, models, and results are created in a temporary folder under the repository root and removed when the tests finish.

```sh
# From the repository root (Python 3.12). tests/test_f1.py needs pyarrow;
# the full suite below also needs numpy.
PYTHONPATH=. python -B -m unittest discover -s tests -p 'test_*.py' -v
```

[tests/test_f1.py](../tests/test_f1.py) builds synthetic parquet files with 3 benchmarks of 101 items each, some with 3 trials. It checks: ALC = 0.25 for a constant prediction, nesting, scoring of all response rows, the hash rule, no leakage into acquisition, the error decomposition, equal weighting of pairs, the cap, anonymization, the split, the function call forms, input copying, prediction reuse, reproducibility, abnormal values and acquisition shortfalls, and running [submission_e1/model.py](../submission_e1/model.py) from the CLI.

## 日本語

### 評価の模擬装置 F1

F1 は、PAIEC（NeurIPS 2026 Predictive AI Evaluation Competition）の streaming の Brier ALC を、手元の公開データで近似する装置です。本番の採点器と同一であるとは主張しません。Brier は確率と実際の 0/1 応答の二乗誤差です。ALC は、獲得ラベル数を増やした際の Brier を所定の重みで平均した値です。小さいほど良い値になります。

### 使い方

リポジトリの直下から実行します。F1 には pyarrow が必要です（Python 3.12 で確認）。標準ライブラリと pyarrow だけを使用し、通信やデータのダウンロードは行いません。

```sh
python -B f1/simulate.py --data /path/to/data --model submission_e1/model.py \
  --benchmarks benchmark_a,benchmark_b --runs 10 --seed 42 \
  --pairs-strategy balanced --split tune --holdout-fraction 0.3 \
  --out results.json
```

`--data` は次の構造を持つディレクトリです。単一 dataset の親ではなく、その dataset 自体を指定することもできます。

```text
data/
  dataset_a/formatted_tables/
    benchmarks.parquet
    subjects.parquet
    items.parquet
    responses.parquet
  dataset_b/formatted_tables/...
```

`--labeling /path/to/labeling.py` を付けると、そのファイルの `acquisition_function` を使います。指定しなければ主催の評価器と同じハッシュ式で抽出します。予測器は `predict(input, labeled)` を公開してください。

| 引数 | 役割・既定値 |
|---|---|
| `--benchmarks a,b` | 元の benchmark_id で対象を限定します。省略すると全候補です。見つからない ID はエラーになります。 |
| `--runs N` / `--seed S` | 繰り返し回数は既定で 1 回です（seed の既定は 0）。各回の seed は `S + 0, S + 1, ...` です。 |
| `--pairs-strategy balanced` | 組を seed 付きで並べ替え、上限を 80 で割った整数までの組を選びます。各組へ 80 item を確保した後、余りを 1 item ずつ巡回配分します。 |
| `--pairs-strategy packed` | 同じ方式で並べ替えた組へ、順に持っている item 数まで配分します。残りが 80 未満になれば終了します。 |
| `--max-sample 1000` | 全組合計の異なる subject–item 数の上限です。80〜1,000 の範囲で縮小できます。反復試行の行数ではありません。 |
| `--split tune\|holdout\|all` | 調整用・確認用・両方の選択です。既定は `all` です。定数を選ぶ際は明示的に `tune` を指定してください。 |
| `--holdout-fraction F` | 確認用に割り当てるハッシュ領域の割合です。既定は 0.2 で、このリポジトリの実験はすべて 0.3 を渡します。実際の件数の比率を保証する値ではありません。 |
| `--release-years 2025,2026` | subject の公開年を限定します。既定では欠損・2025〜2026 年以外を除きます。`all` なら年の制限を外します。 |
| `--out results.json` | 結果の保存先です。親ディレクトリは既存である必要があります。既存ファイルは置き換わります。 |

調整用と確認用は、[splits.py](splits.py) の `benchmark_split` が元の benchmark_id の SHA-256 全 256 bit から決定します。ハッシュ値を `2**256` で割った値が割合未満なら確認用です。run の seed や予測器には依存しません。確認用は定数や特徴を選び終えてから最後に使用してください。F1 は、それ以前に予測器を学習させたベンチマークを自動的には判定しません。

### 資料で確認した部分

ここでの「公式」は 競技の頁 <https://aims.stanford.edu/competition> の 2026-10-17 時点の写し（再配布しません）を指します。Web による追加確認は行っていません。

| 確認した内容 | 根拠 |
|---|---|
| 入力は subject の 8 鍵と item の 4 鍵です。入力値はすべて文字列で、欠損は空文字列です。 | 競技の頁（2026-10-17 の写し）の Submission / input field reference |
| 80 以上の異なる item がある組を対象とし、item 単位で 50/50 に分けます。同じ item の全試行は同じ側に置きます。 | 同頁の Test-time Adaptation |
| 獲得は item の記録済み応答を 1 件だけ明かします。固定のランダム順で候補を提示し、拒否した候補へは戻りません。 | 同頁の Acquisition function |
| 全組の獲得済みラベルを共有します。context は `subject_id`, `benchmark_id`, `labels_acquired`, `labels_remaining`, `max_labels`, `items_remaining` です。残りラベル数は 31 を基準にします。 | 同頁の Acquisition function |
| 0/1/3/7/15/31 の全予算を 1 本の軌跡で評価し、応答の全行を採点します。同じ入力の予測は再利用します。 | 同頁の Test-time Adaptation / Evaluation |
| 組内の応答で Brier を平均した後、ALC を算出し、最後に組を等しく平均します。獲得用と評価用を合わせて最大 1,000 subject–item です。 | 同頁の Computing Brier ALC / Data |
| 既定の抽出式、関数の引数の選び方、深いコピー、厳密な bool 戻り値です。必要な引数形だけ予測値を計算し、関数内部の TypeError は再試行しません。 | [主催の評価器](../tools/official/tools/streaming_ingestion.py) の AcquisitionHook / acquire 分岐 |
| 読み込む表・列の名前と型です。item の `content` を入力の `item_content` に対応させ、`interactors` は responses から取得します。 | 主催の parquet スキーマ（再配布しません） |

このほか F1 は、binary・item の benchmark の選別、公開年の絞り込み、匿名 ID、誤差の分解を実装しています。

### F1 が自分で決めた近似

- **適格性と上限の順番**: 欠損応答を除いて異なる item が 80 以上あるかを先に判定します。その後、組を選び、最大 1,000 件へ制限します。選択後も各組に最低 80 item を残します。80 未満の組は評価しません。
- **組の抽出と配分**: 上記の `balanced` / `packed` の 2 条件です。未使用の枠が残る場合があります。選んだ各組の中では、item を seed 付きで一様に選びます。
- **50/50 の端数**: 奇数なら獲得用を切り捨て、評価用を切り上げます。抽出・分割・候補順・獲得試行はそれぞれ独立の乱数系列で決めます。
- **複数組の進行順**: 元の `(benchmark_id, subject_id)` の辞書順です。全組を各採点点まで進めた後、一斉に評価します。途中の獲得関数は、その時点ですでに進んだ組のラベルも参照できます。
- **匿名化**: 元の ID の SHA-256 全文に `subject_` / `benchmark_` を付けます。主催の匿名 ID そのものは再現しません。item の ID は予測器へ渡しません。結果 JSON の診断用 item ID と benchmark 一覧は元の ID を保持します。
- **繰り返しの試行の選択**: 獲得用 item ごとに、記録済みの有効な応答行から一様に 1 行選びます。trial・interactors・ラベルの順で行を整列してから seed 付きで選ぶため、parquet の行順を変えても選択は安定します。
- **欠損の扱い**: response が null の行は適格性・獲得・採点から除きます。0/1 以外や非有限値はエラーです。文字列以外の入力属性は JSON 文字列にします。
- **候補を使い切った場合**: 獲得関数が拒否し続けた場合、強制的にラベルを購入しません。実際の獲得数を記録し、以後の名目予算では同じラベル集合で評価します。`budget_reached: false` を確認してください。
- **分解の単位**: 通常は item 単位です。同じ item の試行で interactors が異なる場合は、入力ごとにセルを分けます。分割と獲得の単位は引き続き item です。予測値が異なる入力を平均して式へ入れることを避け、観測 Brier との一致を保ちます。
- **繰り返し集計**: 各 run の組平均をさらに平均し、run 間の母標準偏差（`ddof=0`）を出します。1 run の標準偏差は 0 です。
- **実行環境**: 予測器・獲得関数は同じ Python プロセスで実行します。主催の worker 再生成、時間制限、並列処理、隔離環境は再現しません。呼出しごとの入力・ラベルは深いコピーで保護します。

これらの条件と選別理由は結果 JSON の `conditions` と `inventory` に残します。`balanced` と `packed` の双方で、手法の差が維持されるかを比較してください。同じ引数・seed で比べると標本・分割・既定の獲得が一致します。獲得関数自身が乱数や可変状態を使う場合の再現性は、その実装にも依存します。

### 結果と誤差の分解

`summary` に全 run の ALC の平均・標準偏差と、予算別 Brier の平均・標準偏差があります。`runs[].pairs[]` に組別 ALC、標本の件数、獲得用・評価用の item ID を記録します。各 `checkpoints` に Brier、採点した応答行数、獲得済み item ID、実際の獲得数、全組の獲得ラベル数を記録します。モデルへ渡すのは獲得済みの `[input, label]` だけであり、結果 JSON は渡しません。

ALC の重みは `0.1 B0 + 0.2 B1 + 0.2 B3 + 0.2 B7 + 0.2 B15 + 0.1 B31` です。採点用予測は全組で入力を照合し、採点点ごとに 1 回ずつ計算します。ラベルが増える次の採点点では計算し直します。`evaluation_prediction_calls_by_budget` が実呼出し数です。

各 item（必要なら item と入力のセル）に m 個の応答があるとして、`π̂` は応答の標本平均、`w = m / 全応答数` とします。`p̄` と `π̄` は w による平均です。

```text
level_shift           = (p̄ − π̄)²
shape_mismatch        = Σ w ((p − p̄) − (π̂ − π̄))²
empirical_fluctuation = Σ w π̂(1 − π̂)
Brier                 = 上の 3 成分の和
```

実際の応答行から求めた Brier と照合し、差を `decomposition_residual` に記録します。浮動小数点の丸めを考慮し、差の絶対値が `1e-12` を超えれば停止します。この分解の揺らぎは標本上の量です。真の揺らぎの不偏推定ではありません。m=1 ならその item の揺らぎは 0 になります。`m/(m−1)` による補正、較正の診断、AUC は実装していません。

### メモリと限界

parquet は必要な列だけを 8,192 行ずつ読みます。traces・assets・正解の記述を含む grading_criterion は開きません。初めに dataset ごとに適格性を調べ、保持するのは組と件数の一覧です。各 run でも dataset を 1 本ずつ処理し、選ばれた最大 1,000 subject–item の入力・応答だけを全 dataset にわたって保持します。

適格性を調べる間は、その dataset の item ID と組ごとの異なる item ID の集合をメモリに持ちます。1 dataset 自体が非常に大きい場合や、選ばれた item に大量の試行・長文がある場合のメモリ使用量には限界があります。読み直しによってメモリを節約するため、run 数を増やすと I/O が増えます。

本番の非公開データ・標本選択・公開年などの整備基準の全体は再現していません。未見ベンチマークへの性能や本番順位は、合成試験の成功からは分かりません。小さい benchmark への絞り込みによる偏りも残ります。重複 benchmark_id を複数 dataset で検出した場合は、組を二重に数えずエラーにします。

外部の model.py / labeling.py は通常の Python コードとして実行します。通信を封じる隔離装置ではないため、信頼できるローカルのコードだけを指定してください。F1 自体に通信処理はありません。

### 試験

試験用の一時データ・model・結果は、リポジトリの直下の一時フォルダに作り、試験終了後に片付けます。

```sh
# リポジトリの直下から（Python 3.12）。tests/test_f1.py は pyarrow を要し、
# 下の全体の試験は numpy も要る
PYTHONPATH=. python -B -m unittest discover -s tests -p 'test_*.py' -v
```

[tests/test_f1.py](../tests/test_f1.py) は 3 benchmark・各 101 item・一部 3 試行の合成 parquet を作ります。定数予測の ALC=0.25、入れ子、全応答行の採点、ハッシュ式、獲得への漏洩防止、誤差の分解、組の等重み、上限、匿名化、分割、関数の呼出し形式、入力のコピー、予測の再利用、再現性、異常値と獲得不足、[submission_e1/model.py](../submission_e1/model.py) の CLI 実行を確認します。
