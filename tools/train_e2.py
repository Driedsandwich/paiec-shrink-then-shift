"""Offline E2 experiment. Never serializes item content; only numeric results.

Run with PYTHONPATH=. .venv/bin/python -B tools/train_e2.py.
The held-out comparison calls f1.simulate.run_simulation without patching it.
"""

import argparse
from collections import Counter, defaultdict
from functools import partial
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import statistics
import sys
import time

sys.dont_write_bytecode = True

import numpy as np
import pyarrow as pa
from f1 import simulate as sim
from f1.splits import benchmark_split
from variants import e2_features as e2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/e2_item"
YEARS = {"2025", "2026"}
GAMMAS = (0, 0.25, 0.5, 0.75, 1.0)
ALPHAS = (0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)


def rss_mib():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 ** 2 if sys.platform == "darwin" else 1024)


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load_numeric(pairs):
    """Pool trials/eligible subjects per item, preserving release-year/interactor contexts.

    Each item has total weight 1 within its benchmark, divided across its observed
    contexts by response-row frequency. Each benchmark has total weight 1.
    Target = logit((item successes+1)/(item rows+2)) minus that benchmark's
    equal-item mean logit. Subject release years affect features, not targets.
    No texts, individual responses, IDs of subjects, or item features are saved.
    """
    x, targets, weights, bids, item_keys, rates = [], [], [], [], [], []
    audit = []
    for folder in sorted({p.folder for p in pairs}):
        wanted = {p.key for p in pairs if p.folder == folder}
        subjects, _ = sim.subjects_in(folder, YEARS)
        stats = {}
        contexts = defaultdict(Counter)
        for row in sim.read_rows(folder, "responses", sim.RESPONSE_COLUMNS):
            if (row["benchmark_id"], row["subject_id"]) not in wanted:
                continue
            y = sim.response_label(row)
            if y is None:
                continue
            key = row["benchmark_id"], row["item_id"]
            s, n = stats.get(key, (0, 0))
            stats[key] = s + y, n + 1
            year = subjects[row["subject_id"]]["release_date"][:4]
            contexts[key][year, sim.text_value(row["interactors"])] += 1
        by_benchmark = defaultdict(list)
        for (bid, iid), (s, n) in stats.items():
            p = (s + 1) / (n + 2)
            by_benchmark[bid].append(math.log(p / (1 - p)))
        centers = {bid: statistics.mean(v) for bid, v in by_benchmark.items()}
        for row in sim.read_rows(folder, "items", ("item_id", "benchmark_id", "content", "item_features")):
            key = row["benchmark_id"], row["item_id"]
            if key not in stats:
                continue
            bid, iid = key
            s, n = stats[key]
            rate = (s + 1) / (n + 2)
            target = math.log(rate / (1 - rate)) - centers[bid]
            for (year, interactors), count in sorted(contexts[key].items()):
                item = {"item_content": sim.text_value(row["content"]),
                        "item_features": sim.text_value(row["item_features"]),
                        "interactors": interactors, "benchmark_id": sim.anonymous("benchmark", bid)}
                x.append(e2.feature_vector(item, {"release_date": year}))
                targets.append(target)
                weights.append(count / n / len(by_benchmark[bid]))
                bids.append(bid)
                item_keys.append((bid, iid))
                rates.append(s / n)
        for bid, values in sorted(by_benchmark.items()):
            audit.append({"benchmark_id": bid, "items": len(values),
                          "response_rows": sum(n for (b, _), (_, n) in stats.items() if b == bid),
                          "target_center_logit": centers[bid]})
    return {"x": np.asarray(x, dtype=float), "y": np.asarray(targets),
            "w": np.asarray(weights), "bids": np.asarray(bids), "item_keys": item_keys,
            "rates": rates, "audit": audit}


def fit(x, y, weights, alpha):
    w = weights / weights.sum()
    means = np.sum(w[:, None] * x, axis=0)
    scales = np.sqrt(np.sum(w[:, None] * (x - means) ** 2, axis=0))
    scales[scales < 1e-10] = 1.0
    z = (x - means) / scales
    intercept = float(np.sum(w * y))
    coefficients = np.linalg.solve((z * w[:, None]).T @ z + alpha * np.eye(x.shape[1]),
                                   z.T @ (w * (y - intercept)))
    prediction = z @ coefficients + intercept
    return {"feature_names": list(e2.FEATURE_NAMES), "means": means.tolist(),
            "scales": scales.tolist(), "weights": coefficients.tolist(),
            "intercept": intercept, "global_mean": float(np.sum(w * prediction)),
            "alpha": alpha, "scan_chars": e2.SCAN_CHARS}


def predict_numeric(x, params):
    return ((x - params["means"]) / params["scales"]) @ np.asarray(params["weights"]) + params["intercept"]


def cross_validate(data):
    groups = sorted(set(data["bids"]))
    folds = {a: [] for a in ALPHAS}
    for bid in groups:
        validation = data["bids"] == bid
        training = ~validation
        for alpha in ALPHAS:
            params = fit(data["x"][training], data["y"][training], data["w"][training], alpha)
            error = predict_numeric(data["x"][validation], params) - data["y"][validation]
            mse = float(np.average(error ** 2, weights=data["w"][validation]))
            folds[alpha].append({"benchmark_id": bid, "mse": mse})
    rows = [{"alpha": a, "mean_benchmark_mse": statistics.mean(f["mse"] for f in ff),
             "folds": ff} for a, ff in folds.items()]
    best = min(rows, key=lambda row: (row["mean_benchmark_mse"], -row["alpha"]))
    return fit(data["x"], data["y"], data["w"], best["alpha"]), rows


def describe(runs):
    return {"alc_mean": statistics.mean(r["alc"] for r in runs),
            "alc_std": statistics.pstdev(r["alc"] for r in runs),
            "brier_by_budget": {str(b): {
                "mean": statistics.mean(r["brier_by_budget"][str(b)] for r in runs),
                "std": statistics.pstdev(r["brier_by_budget"][str(b)] for r in runs)}
                for b in sim.BUDGETS}}


def paired(runs, base):
    assert [r["seed"] for r in runs] == [r["seed"] for r in base]
    diffs = [r["alc"] - b["alc"] for r, b in zip(runs, base)]
    return {"mean": statistics.mean(diffs),
            "standard_error": statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0,
            "n": len(diffs), "differences": diffs}


def tune_gamma(pairs, parameters):
    """Full-content F1 acquisition; algebraic Brier replay for five gammas.

    No patching of F1. The replay is checked against run_simulation below.
    Text and individual feature vectors remain in memory and are discarded.
    """
    runs = {g: [] for g in GAMMAS}
    audits = []
    for seed in range(20):
        states = sim.prepare_states(sim.allocate_pairs(pairs, seed, "balanced", 1000), seed, YEARS)
        labeled = []
        acquired_scores = defaultdict(list)
        evaluation = {}
        for state in states:
            evaluation[state.pair] = [(input, ys, e2.feature_score(input[1], input[0], parameters))
                                      for _, input, ys in state.evaluation]
        briers = {g: {} for g in GAMMAS}
        digest = hashlib.sha256()
        for budget in sim.BUDGETS:
            for state in states:
                while len(state.acquired_ids) < budget and state.cursor < len(state.acquisition):
                    iid, input, y = state.acquisition[state.cursor]
                    context = {"subject_id": sim.anonymous("subject", state.pair.subject),
                               "benchmark_id": sim.anonymous("benchmark", state.pair.benchmark),
                               "labels_acquired": len(state.acquired_ids),
                               "labels_remaining": 31 - len(state.acquired_ids), "max_labels": 31,
                               "items_remaining": len(state.acquisition) - state.cursor}
                    state.cursor += 1
                    if sim.default_acquire(input, context):
                        state.acquired_ids.append(iid)
                        labeled.append([input, y])
                        acquired_scores[input[1]["benchmark_id"]].append(
                            e2.feature_score(input[1], input[0], parameters))
            pair_scores = {g: [] for g in GAMMAS}
            for state in states:
                cells = evaluation[state.pair]
                p = e2.e1b(cells[0][0], labeled)
                scores = acquired_scores[cells[0][0][1]["benchmark_id"]]
                center = statistics.mean(scores) if scores else parameters["global_mean"]
                total = sum(len(ys) for _, ys, _ in cells)
                for gamma in GAMMAS:
                    loss = 0.0
                    for _, ys, score in cells:
                        q = p if gamma == 0 else e2.clip(e2.sigmoid(math.log(p / (1 - p)) + gamma * (score - center)))
                        loss += len(ys) * q * q - 2 * q * sum(ys) + sum(ys)
                    pair_scores[gamma].append(loss / total)
                digest.update(sim.canonical([budget, state.pair.key, state.acquired_ids]).encode())
            for gamma in GAMMAS:
                briers[gamma][str(budget)] = statistics.mean(pair_scores[gamma])
        for gamma in GAMMAS:
            bb = briers[gamma]
            runs[gamma].append({"seed": seed, "brier_by_budget": bb,
                                "alc": math.fsum(w * bb[str(b)] for b, w in zip(sim.BUDGETS, sim.WEIGHTS))})
        audits.append({"seed": seed, "pairs": len(states), "trajectory_sha256": digest.hexdigest()})
        if seed % 5 == 4:
            print(f"Gamma replay: {seed + 1}/20 seeds; RSS={rss_mib():.1f} MiB", flush=True)
    grid = [{"gamma": g, "summary": describe(rr), "paired_vs_e1b": paired(rr, runs[0]), "runs": rr}
            for g, rr in runs.items()]
    best = min(grid, key=lambda row: (row["summary"]["alc_mean"], row["gamma"]))
    return best["gamma"], grid, audits


def load_model():
    spec = importlib.util.spec_from_file_location("submission_e2_local", ROOT / "submission_e2/model.py")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    return model


def emit_submission(parameters, gamma, benchmarks):
    folder = ROOT / "submission_e2"
    folder.mkdir(exist_ok=True)
    source = (ROOT / "variants/e2_features.py").read_text()
    source += "\n\nPARAMETERS = " + repr(parameters) + "\nGAMMA = " + repr(gamma)
    source += "\n\ndef predict(input, labeled=None):\n    return predict_with_parameters(input, labeled, PARAMETERS, GAMMA)\n"
    (folder / "model.py").write_text(source)
    (folder / "README.md").write_text(
        "# E2: 問題の相対正答しやすさによる補正\n\n"
        "E1b の階層的な平滑化確率に、問題特徴から推定した相対正答しやすさをロジット尺度で加えます。\n"
        "式は sigmoid(logit(E1b) + γ × (f(x) − c)) です。c は同一 benchmark_id の有効な取得済みラベル入力の f の平均です。\n"
        "ラベルがない場合は学習時の全体平均を使います。標準ライブラリだけで動作し、呼出し間の状態を保存しません。\n"
        "失敗時は E1b を返し、出力を [0.001, 0.999] に制限します。獲得関数は指定せず既定の無作為獲得を使います。\n\n"
        f"γ={gamma}、ridge 正則化={parameters['alpha']}。重み、平均、尺度は model.py 内の辞書に埋め込みました。\n"
        "特徴は全文字数の対数、推定行数の対数、数字率、ASCII 英字以外の率、zlib 圧縮率、コード・数式・選択肢の印、\n"
        "URL、interactors の有無、item_features の最上位キー数、本文の最大年と release_date 年の差、年の欠損指標です。\n"
        f"特徴走査は最大 {e2.SCAN_CHARS} 文字の先頭・末尾です。文字数だけは本文全体を数えます。本文は記録しません。\n\n"
        "学習データは measurement-db revision 54074453（依頼者指定）です。binary/item、2025〜2026 年公開 subject、80 問以上の組を使いました。\n"
        "benchmark_split(id, 0.3)==tune のベンチマークだけから、問題の平滑化正答率のロジットをベンチマーク内で中心化して学習しました。\n"
        "各ベンチマークは同じ重み、ベンチマーク内は各問題を同じ重みとしました。release 年・interactors が異なる入力は応答行比率で分けました。\n"
        "正則化は調整用内でベンチマークを一つずつ除く交差検証、γ は調整用 seeds 0〜19 の ALC で選びました。\n"
        "確認用 seeds 1000〜1049 は選択に使っていません。ベンチマーク識別子やモデル名自体は特徴にしません。\n\n"
        "学習に用いた調整用ベンチマーク:\n\n" + "\n".join("- " + b for b in benchmarks) +
        "\n\nLLM（大規模言語モデル）によるコーディング支援として OpenAI Codex を使用しました。方法・実験・コードの作成支援を開示します。\n")


def slim(result):
    """Retain aggregate/run metrics only; never write inputs or response contents."""
    return {"summary": result["summary"], "conditions": result["conditions"],
            "eligible_pair_count": result["eligible_pair_count"],
            "runs": [{"seed": r["seed"], "alc": r["alc"], "brier_by_budget": r["brier_by_budget"],
                      "pairs": len(r["pairs"]), "sampled_unique_subject_items": r["sampled_unique_subject_items"]}
                     for r in result["runs"]]}


def rank_values(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2
        i = j
    return ranks


def rank_correlation(data, parameters):
    scores = predict_numeric(data["x"], parameters)
    items = defaultdict(lambda: [0.0, 0.0, 0.0])
    for key, score, weight, rate in zip(data["item_keys"], scores, data["w"], data["rates"]):
        cell = items[key]
        cell[0] += float(score * weight)
        cell[1] += float(weight)
        cell[2] = rate
    grouped = defaultdict(list)
    for (bid, _), (ss, w, rate) in items.items():
        grouped[bid].append((ss / w, rate))
    rows = []
    for bid, cells in sorted(grouped.items()):
        xx, yy = map(rank_values, zip(*cells))
        rho = float(np.corrcoef(xx, yy)[0, 1]) if np.std(xx) > 0 and np.std(yy) > 0 else None
        rows.append({"benchmark_id": bid, "items": len(cells), "spearman_rho": rho,
                     "difficulty_vs_success_spearman": -rho if rho is not None else None})
    valid = [r["spearman_rho"] for r in rows if r["spearman_rho"] is not None]
    return {"mean_benchmark_spearman": statistics.mean(valid) if valid else None,
            "mean_difficulty_vs_success_spearman": -statistics.mean(valid) if valid else None,
            "valid_benchmarks": len(valid), "by_benchmark": rows,
            "orientation": "f predicts ease (higher success probability); difficulty=-f reverses the sign",
            "population": "all eligible holdout items; pooled observed response rates, average ranks for ties"}


def latency(model):
    # Synthetic text, not measurement-db content. Include maximal allowed evidence.
    long_item = {"item_content": ("Synthetic task 2024. def demo(): return 1;\n" * 25000),
                 "item_features": "{}", "interactors": "", "benchmark_id": "synthetic"}
    subject = {"release_date": "2025-01", "normalized_name": "synthetic"}
    labels = [[[dict(subject, normalized_name=str(i)),
                dict(long_item, item_content=((f"Synthetic {i:04d}. def demo(): return 1;\n" * 220) + str(i)))],
               i % 2] for i in range(372)]
    rows = []
    for n in (0, 31, 372):
        samples = []
        for _ in range(25):
            start = time.perf_counter()
            p = model.predict([subject, long_item], labels[:n])
            samples.append(time.perf_counter() - start)
            assert 0.001 <= p <= 0.999
        rows.append({"body_chars": len(long_item["item_content"]), "labeled_entries": n,
                     "labeled_distinct_items": n, "labeled_body_chars_min": min(
                         (len(entry[0][1]["item_content"]) for entry in labels[:n]), default=0),
                     "calls": len(samples), "median_seconds": statistics.median(samples),
                     "p95_seconds": sorted(samples)[math.ceil(.95 * len(samples)) - 1],
                     "maximum_seconds": max(samples)})
    return rows


def summary(result):
    h = result["holdout"]
    d = h["paired_e2_minus_e1b"]
    lines = ["# E2 問題特徴の学習・確認実験", "",
             result["recommendation"], "",
             "ALC は予算 0/1/3/7/15/31 の Brier を 0.1/0.2/0.2/0.2/0.2/0.1 で平均し、subject と benchmark の組を等しく平均した損失です。小さいほど良い値です。",
             "標準偏差は seed ごとの組平均の母標準偏差です。対の差は同じ seed の E2−E1b、標準誤差は差の標本標準偏差÷√50 です。", "",
             "## 確認用（本文を含む run_simulation、balanced、50 seeds）", "",
             "| 手法 | ALC 平均 ± 標準偏差 |", "|---|---:|"]
    for name in ("E1b", "E2"):
        s = h[name]["summary"]
        lines.append(f"| {name} | {s['alc_mean']:.8f} ± {s['alc_std']:.8f} |")
    lines += ["", f"対の差は {d['mean']:+.8f} ± {d['standard_error']:.8f}（標準誤差）です。", "",
              "| 手法 | 予算0 | 予算1 | 予算3 | 予算7 | 予算15 | 予算31 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("E1b", "E2"):
        lines.append("| " + name + " | " + " | ".join(f"{h[name]['summary']['brier_by_budget'][str(b)]['mean']:.8f}" for b in sim.BUDGETS) + " |")
    lines += ["", "## 特徴と選択", "",
              f"正則化 α={result['parameters']['alpha']}、γ={result['gamma']} です。正則化は調整用内のベンチマーク単位の leave-one-out 交差検証で選びました。γ は調整用 seeds 0〜19 の平均 ALC 最小値で選び、確認用では変更していません。", "",
              "| γ | 調整用 ALC | E1b との差 ± 標準誤差 |", "|---:|---:|---:|"]
    for row in result["gamma_grid"]:
        p = row["paired_vs_e1b"]
        lines.append(f"| {row['gamma']} | {row['summary']['alc_mean']:.8f} | {p['mean']:+.8f} ± {p['standard_error']:.8f} |")
    lines += ["", "標準化した特徴の重みです。正は正答しやすい方向、負は難しい方向を表します。", "",
              "| 特徴 | 重み | 学習平均 | 学習尺度 |", "|---|---:|---:|---:|"]
    p = result["parameters"]
    rows = sorted(zip(p["feature_names"], p["weights"], p["means"], p["scales"]), key=lambda v: -abs(v[1]))
    for name, w, m, s in rows:
        lines.append(f"| {name} | {w:+.8f} | {m:.8f} | {s:.8f} |")
    lines += ["", "## 順位相関（確認用の全適格問題）", "",
              "Spearman の順位相関を各ベンチマークで計算して等しく平均しました。同順位は平均順位です。f は正答しやすさを表し、実際の正答率との正の相関を期待します。難しさを −f と定義すると符号は反転します。", "",
              "| ベンチマーク | 問題数 | f と正答率の順位相関 | −f（難しさ）と正答率の順位相関 |", "|---|---:|---:|---:|"]
    for row in result["rank_correlation"]["by_benchmark"]:
        rho = row["spearman_rho"]
        negative = row["difficulty_vs_success_spearman"]
        lines.append(f"| {row['benchmark_id']} | {row['items']} | {rho if rho is None else f'{rho:+.6f}'} | {negative if negative is None else f'{negative:+.6f}'} |")
    lines += ["", f"f と正答率の相関の平均: {result['rank_correlation']['mean_benchmark_spearman']:+.6f}。難しさ −f では {result['rank_correlation']['mean_difficulty_vs_success_spearman']:+.6f} です。", "",
              "## 学習・条件", "",
              "目的値は各問題の (成功数+1)/(応答数+2) のロジットから、同じベンチマークの問題を等しく平均したロジットを引いた値です。80 問以上の組の 2025〜2026 年 subject の応答だけを使います。各ベンチマークに同じ重み、各問題に同じ重みを付けます。release 年と interactors が違う文脈には応答行数に比例して問題の重みを分けます。",
              "特徴の標準化は各交差検証の学習側だけで計算します。ベンチマーク名・subject 名は回帰に使いません。E1b の事前平均は指定値を固定しました。",
              f"特徴は先頭・末尾の合計 {e2.SCAN_CHARS} 文字を走査します。全文字数は正確で、行数は走査部分から推定します。年は 1000〜2199、年差は ±100 に制限します。省略された本文の中央の印や年は検出しません。",
              f"データ revision: {result['data_revision']}（依頼者指定・通信での照合は未実施）。holdout_fraction=0.3、各組80問以上、1 seed 上限1000 subject–item。", "",
              "調整用: " + ", ".join(result["assignments"]["tune"]) + "。", "",
              "確認用: " + ", ".join(result["assignments"]["holdout"]) + "。", "",
              "## 実行・検証", "",
              f"- 実験: `{result['command']}`。単一プロセス、Arrow/BLAS は各1スレッドです。",
              f"- 実験時間 {result['resources']['elapsed_seconds']:.3f} 秒、最大常駐メモリ {result['resources']['max_rss_mib']:.3f} MiB。",
              f"- 調整用の高速集計と元の模擬装置の照合: {result['parity_checks']}。",
              "- 確認用の両手法は改変していない run_simulation(runs=50, seed=1000, split='holdout', holdout_fraction=0.3, pairs_strategy='balanced') を直接呼びました。本文の空文字への置換はありません。", "",
              "| 本文文字数 | 取得済み件数 | 呼出し回数 | 中央値ms | 95百分位ms | 最大ms |", "|---:|---:|---:|---:|---:|---:|"]
    for row in result["latency"]:
        lines.append(f"| {row['body_chars']} | {row['labeled_entries']} | {row['calls']} | {1000*row['median_seconds']:.3f} | {1000*row['p95_seconds']:.3f} | {1000*row['maximum_seconds']:.3f} |")
    if "prediction_timing" in h["E2"]:
        timing = h["E2"]["prediction_timing"]
        lines += ["", f"- 実データの予測 {timing['calls']} 回: 中央値 {timing['median_seconds']*1000:.3f} ms、95百分位 {timing['p95_seconds']*1000:.3f} ms、最大 {timing['maximum_seconds']*1000:.3f} ms。本文を含み、模擬装置の deepcopy は計時外です。"]
    if "reuse_provenance" in h["E1b"]:
        lines += ["- E1b は同じ作業内で先に完了した本文あり50 seedsを再利用しました。高速化前の E2 測定は中断し、そのスコアは採用していません。学習とγ選択は最終実装で再実行しました。"]
    if "verification" in result:
        v = result["verification"]
        lines += ["", f"- 試験: {v['tests']['tests_run']} 件、失敗 {v['tests']['failures']} 件、エラー {v['tests']['errors']} 件。",
                  f"- 試験命令: `{v['tests']['command']}`。既存試験の一時出力先だけを実行時に許可範囲へ移し、同じ unittest discover 集合を実行しました。",
                  f"- zip 作成命令: `{v['build_command']}`。",
                  f"- zip 検査命令: `{v['zip_check_command']}`。-B は読取専用領域へのバイトコード生成を抑止します。",
                  f"- zip 検査: {v['zip_check']}。",
                  f"- 検証を含む全体経過: {v['total_wall_seconds']:.3f} 秒（最初の時刻記録から）。最大観測プロセスRSS {v['maximum_observed_process_rss_mib']:.3f} MiB。上の実験時間とメモリは最終実行だけの値です。"]
    lines += ["", "## 限界", ""] + ["- " + text for text in result["limitations"]]
    return "\n".join(lines) + "\n"


def run_tests():
    """Same unittest discovery; relocate existing tests' temporary writes only."""
    import tempfile
    import unittest
    from unittest.mock import patch
    from tools import tune_e1
    OUT.mkdir(parents=True, exist_ok=True)
    original = tempfile.TemporaryDirectory
    def temporary(*args, **kwargs):
        target = Path(kwargs.get("dir") or "/")
        if target == ROOT or target == ROOT / "experiments/e1_tuning":
            kwargs["dir"] = OUT
        return original(*args, **kwargs)
    start = time.monotonic()
    with patch.object(tempfile, "TemporaryDirectory", temporary), patch.object(tune_e1, "OUT", OUT):
        test_result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py"))
    record = {"tests_run": test_result.testsRun, "failures": len(test_result.failures),
              "errors": len(test_result.errors), "successful": test_result.wasSuccessful(),
              "elapsed_seconds": time.monotonic() - start, "max_rss_mib": rss_mib(),
              "command": "PYTHONPATH=. .venv/bin/python -B tools/train_e2.py --test-suite",
              "discovery": "unittest discover -s tests -p 'test_*.py'",
              "temporary_write_redirect": "existing tests ROOT/e1_tuning destinations -> experiments/e2_item; no test logic altered"}
    dump(OUT / "tests.json", record)
    return 0 if test_result.wasSuccessful() else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT.parent / "_data/mdb")
    parser.add_argument("--test-suite", action="store_true")
    parser.add_argument("--reuse-e1b", action="store_true",
                        help="reuse this experiment's completed full-content E1b 50-seed result")
    args = parser.parse_args(argv)
    if args.test_suite:
        return run_tests()
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    all_pairs, _ = sim.inventory(args.data, None, "all", 0.3, YEARS)
    pairs = {split: [p for p in all_pairs if benchmark_split(p.benchmark, 0.3) == split]
             for split in ("tune", "holdout")}
    assignments = {split: sorted({p.benchmark for p in pp}) for split, pp in pairs.items()}
    print(f"Inventory: tune={len(pairs['tune'])} pairs/{len(assignments['tune'])} benchmarks; "
          f"holdout={len(pairs['holdout'])} pairs/{len(assignments['holdout'])} benchmarks", flush=True)
    train = load_numeric(pairs["tune"])
    print(f"Numeric training: {len(train['y'])} contexts; RSS={rss_mib():.1f} MiB", flush=True)
    parameters, cv = cross_validate(train)
    dump(OUT / "weights.json", parameters)
    print(f"Ridge selected alpha={parameters['alpha']}", flush=True)
    gamma, grid, audits = tune_gamma(pairs["tune"], parameters)
    emit_submission(parameters, gamma, assignments["tune"])
    model = load_model()
    e1b = sim.load_function(ROOT / "submission_e1b/model.py", "predict")
    parity = []
    for g in sorted({0, gamma}):
        reference = sim.run_simulation(args.data, partial(e2.predict_with_parameters, parameters=parameters, gamma=g),
                                       runs=1, seed=0, split="tune", holdout_fraction=0.3, pairs_strategy="balanced")
        replay = next(row for row in grid if row["gamma"] == g)["runs"][0]
        error = max(abs(replay["brier_by_budget"][str(b)] - reference["runs"][0]["brier_by_budget"][str(b)])
                    for b in sim.BUDGETS)
        if error > 1e-12:
            raise ArithmeticError("Full-content gamma replay parity failed")
        parity.append({"gamma": g, "maximum_brier_error": error})
    holdout = {}
    for name, function in (("E1b", e1b), ("E2", model.predict)):
        if name == "E1b" and args.reuse_e1b:
            cache_path = OUT / "e1b_holdout.json"
            cached = json.loads(cache_path.read_text())
            c = cached["conditions"]
            if not (c["data"] == str(args.data.resolve()) and c["split"] == "holdout"
                    and c["holdout_fraction"] == .3 and c["pairs_strategy"] == "balanced"
                    and c["runs"] == 50 and c["seed"] == 1000
                    and c["max_unique_subject_items"] == 1000
                    and c["release_years"] == sorted(YEARS)
                    and [r["seed"] for r in cached["runs"]] == list(range(1000, 1050))):
                raise ValueError("E1b cached conditions differ")
            cached["reuse_provenance"] = {"file_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                "source": "completed full-content E1b measurement of this same experiment; train.log",
                "model_sha256": hashlib.sha256((ROOT / "submission_e1b/model.py").read_bytes()).hexdigest()}
            holdout[name] = cached
            print("Reuse completed full-content E1b 50-seed measurement (same conditions)", flush=True)
            continue
        phase = time.monotonic()
        prediction_times = []
        def measured_predict(input, labeled):
            start = time.perf_counter()
            value = function(input, labeled)
            prediction_times.append(time.perf_counter() - start)
            if len(prediction_times) % 25000 == 0:
                print(f"{name}: {len(prediction_times)} predictions; elapsed={time.monotonic()-phase:.1f}s", flush=True)
            return value
        print(f"Real run_simulation: {name}, holdout seeds 1000..1049, full content", flush=True)
        holdout[name] = slim(sim.run_simulation(args.data, measured_predict, runs=50, seed=1000,
                             split="holdout", holdout_fraction=0.3, pairs_strategy="balanced"))
        holdout[name]["prediction_timing"] = {
            "calls": len(prediction_times), "mean_seconds": statistics.mean(prediction_times),
            "median_seconds": statistics.median(prediction_times),
            "p95_seconds": float(np.quantile(prediction_times, .95)), "maximum_seconds": max(prediction_times),
            "scope": "actual full-content predict calls; excludes simulator deepcopy; timer only, no input modifications"}
        holdout[name]["elapsed_seconds"] = time.monotonic() - phase
        dump(OUT / (name.lower() + "_holdout.json"), holdout[name])
        print(f"{name}: ALC={holdout[name]['summary']['alc_mean']:.8f}; seconds={time.monotonic()-phase:.1f}; "
              f"RSS={rss_mib():.1f} MiB", flush=True)
    difference = paired(holdout["E2"]["runs"], holdout["E1b"]["runs"])
    holdout["paired_e2_minus_e1b"] = difference
    rank = rank_correlation(load_numeric(pairs["holdout"]), parameters)
    timings = latency(model)
    improvement = gamma > 0 and difference["mean"] < -2 * difference["standard_error"]
    recommendation = ("E2 を提出候補に推します。確認用の平均 ALC が E1b より低く、改善幅が対の差の標準誤差の2倍を上回りました。"
                      if improvement else
                      "提出は E1b を推します。E2 の確認用改善が対の差の標準誤差の2倍を上回る条件を満たさず、追加補正を採用する根拠が十分ではありません。")
    result = {"format": "x053_e2_item_v1",
              "data_revision": "54074453", "data_revision_source": "user supplied; no network access",
              "assignments": assignments,
              "conditions": {"holdout_fraction": 0.3, "release_years": sorted(YEARS), "minimum_items": 80,
                  "max_sample": 1000, "pairs_strategy": "balanced", "tune_seeds": list(range(20)),
                  "holdout_seeds": list(range(1000,1050)), "processes": 1, "arrow_threads": 1,
                  "content_read": True, "content_replaced": False, "eligible_pairs": len(all_pairs),
                  "equal_benchmark_training_weights": True, "cv": "leave-one-benchmark-out; fold-specific standardization"},
              "training": {"contexts": len(train["y"]), "benchmarks": train["audit"],
                           "target_smoothing": "(pooled item successes+1)/(pooled item response rows+2)",
                           "centering": "equal-item benchmark mean of smoothed logits",
                           "context_weight": "response proportion within item; equal items within benchmark; equal benchmarks"},
              "parameters": parameters, "ridge_cv": cv, "gamma": gamma, "gamma_grid": grid,
              "tune_trajectories": audits, "parity_checks": parity, "holdout": holdout,
              "rank_correlation": rank, "latency": timings, "recommend_e2": improvement,
              "recommendation": recommendation,
              "command": "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -B tools/train_e2.py" + (" --reuse-e1b" if args.reuse_e1b else ""),
              "limitations": ["ローカル模擬装置の結果であり、本番との同一性や未公開ベンチマークへの一般化は未確認です。",
                  "50 seeds は同じ有限データの再標本化です。差の標準誤差は未知のベンチマーク間の不確実性を表しません。",
                  "本文の走査上限により中央の構造を見落とします。特徴は浅い統計量で、問題の意味は理解しません。",
                  "問題別正答率は subject・試行をまとめた観測値です。subject の構成と試行数の偏りが残ります。",
                  "事前平均は E1b の指定値を固定しました。前回の本文なし実験のスコアとは獲得経路が異なるため直接比較できません。",
                  "γ の調整用評価は特徴学習と同じベンチマークを使います。交差検証は ridge 強度の選択だけであり、γ の完全な入れ子評価ではありません。",
                  "順位相関は確認用の全適格問題の補正前 f と観測正答率の相関です。ALC の50 seedsだけの問題集合でも最終予測確率の相関でもありません。",
                  "遅延は合成長文と互いに異なる取得済み入力で測り、モジュール読み込みや模擬装置による deepcopy は含めません。実行環境による揺らぎがあります。"],
              "resources": {"elapsed_seconds": time.monotonic()-started, "max_rss_mib": rss_mib()}}
    dump(OUT / "results.json", result)
    (OUT / "summary.md").write_text(summary(result))
    print(json.dumps({"gamma": gamma, "difference": {k:v for k,v in difference.items() if k != "differences"},
                      "rank_correlation": rank["mean_benchmark_spearman"], "resources": result["resources"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
