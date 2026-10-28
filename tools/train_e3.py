"""Offline E3 tuning, full-content F1 confirmation, and submission generation.

Only numeric aggregates and benchmark names are serialized. No input content,
subject identities, individual labels, or input hashes are written. Single
process; Arrow threads are limited to one. No network access is performed.
"""

import argparse
from collections import Counter, defaultdict
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

import pyarrow as pa
from f1 import simulate as sim
from variants import e3_same_item as e3
from submission_e2 import model as e2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/e3_item"
DATA = ROOT.parent / "_data/mdb"
YEARS = {"2025", "2026"}
LAMBDAS = (0.5, 1, 2, 4, 8)
ETAS = (0.5, 1.0)
MODES = ("probability", "logit")
COMPONENTS = ("level_shift", "shape_mismatch", "empirical_fluctuation")


def rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load_model(path):
    spec = importlib.util.spec_from_file_location("e3_local_" + path.parent.name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def describe(runs):
    return {"alc_mean": statistics.mean(r["alc"] for r in runs),
            "alc_std": statistics.pstdev(r["alc"] for r in runs),
            "brier_by_budget": {str(b): {
                "mean": statistics.mean(r["brier_by_budget"][str(b)] for r in runs),
                "std": statistics.pstdev(r["brier_by_budget"][str(b)] for r in runs)} for b in sim.BUDGETS}}


def paired(runs, base):
    assert [r["seed"] for r in runs] == [r["seed"] for r in base]
    diffs = [r["alc"] - b["alc"] for r, b in zip(runs, base)]
    return {"mean": statistics.mean(diffs),
            "standard_error": statistics.stdev(diffs) / math.sqrt(len(diffs)),
            "better_seeds": sum(d < 0 for d in diffs), "equal_seeds": sum(d == 0 for d in diffs),
            "worse_seeds": sum(d > 0 for d in diffs), "n": len(diffs), "differences": diffs}


def tune(data):
    """Replay all candidates on one full-content default acquisition trajectory.

    E2 feature scores are computed once per input, and residual statistics once
    per evaluated input/checkpoint. Only the final scalar addition differs among
    the 20 candidates. A direct run_simulation parity check follows selection.
    """
    started = time.monotonic()
    pairs, _ = sim.inventory(data, None, "tune", 0.3, YEARS)
    configs = [dict(shrinkage=l, eta=e, mode=m) for m in MODES for l in LAMBDAS for e in ETAS]
    runs = [[] for _ in configs]
    baseline_runs = []
    for seed in range(20):
        states = sim.prepare_states(sim.allocate_pairs(pairs, seed, "balanced", 1000), seed, YEARS)
        labeled, acquired_scores = [], defaultdict(list)
        evaluation = {s.pair: [(x, ys, e2.feature_score(x[1], x[0], e2.PARAMETERS))
                               for _, x, ys in s.evaluation] for s in states}
        briers = [dict() for _ in configs]
        base_briers = {}
        for budget in sim.BUDGETS:
            for state in states:
                while len(state.acquired_ids) < budget and state.cursor < len(state.acquisition):
                    iid, x, y = state.acquisition[state.cursor]
                    context = {"subject_id": sim.anonymous("subject", state.pair.subject),
                               "benchmark_id": sim.anonymous("benchmark", state.pair.benchmark),
                               "labels_acquired": len(state.acquired_ids),
                               "labels_remaining": 31 - len(state.acquired_ids), "max_labels": 31,
                               "items_remaining": len(state.acquisition) - state.cursor}
                    state.cursor += 1
                    if sim.default_acquire(x, context):
                        state.acquired_ids.append(iid)
                        labeled.append([x, y])
                        acquired_scores[x[1]["benchmark_id"]].append(e2.feature_score(x[1], x[0], e2.PARAMETERS))
            per_pair = [[] for _ in configs]
            base_pair = []
            for state in states:
                cells = evaluation[state.pair]
                p = e2.e1b(cells[0][0], labeled)
                scores = acquired_scores[cells[0][0][1]["benchmark_id"]]
                center = math.fsum(scores) / len(scores) if scores else e2.PARAMETERS["global_mean"]
                losses = [[] for _ in configs]
                base_losses = []
                total = sum(len(ys) for _, ys, _ in cells)
                for x, ys, score in cells:
                    baseline = e2.clip(e2.sigmoid(math.log(p / (1-p)) + e2.GAMMA * (score-center)))
                    stats = e3.same_item_stats(x, labeled)
                    count, success = len(ys), sum(ys)
                    base_losses.append(count * baseline**2 - 2 * baseline * success + success)
                    for j, config in enumerate(configs):
                        q = e3.apply_shift(baseline, stats, **config)
                        losses[j].append(count * q**2 - 2 * q * success + success)
                base_pair.append(math.fsum(base_losses) / total)
                for j in range(len(configs)):
                    per_pair[j].append(math.fsum(losses[j]) / total)
            base_briers[str(budget)] = statistics.mean(base_pair)
            for j in range(len(configs)):
                briers[j][str(budget)] = statistics.mean(per_pair[j])
        for rr, bb in zip(runs + [baseline_runs], briers + [base_briers]):
            rr.append({"seed": seed, "brier_by_budget": bb,
                       "alc": math.fsum(w * bb[str(b)] for b, w in zip(sim.BUDGETS, sim.WEIGHTS))})
        print(f"Tune {seed+1}/20; RSS {rss_mib():.1f} MiB", flush=True)
    grid = [{"config": c, "summary": describe(rr), "runs": rr,
             "paired_vs_e2": paired(rr, baseline_runs)} for c, rr in zip(configs, runs)]
    best = min(grid, key=lambda row: (row["summary"]["alc_mean"], row["config"]["mode"],
                                     -row["config"]["shrinkage"], row["config"]["eta"]))
    result = {"config": best["config"], "grid": grid, "E2": {"summary": describe(baseline_runs),
              "runs": baseline_runs}, "benchmarks": sorted({p.benchmark for p in pairs}),
              "eligible_pairs": len(pairs), "seconds": time.monotonic() - started,
              "max_rss_mib": rss_mib()}
    dump(OUT / "tune.json", result)
    return result


def emit_submission(config, benchmarks):
    folder = ROOT / "submission_e3"
    folder.mkdir(exist_ok=True)
    source = (ROOT / "submission_e2/model.py").read_text()
    source += "\n\n# Frozen E2 fallback; no additional model files are needed.\npredict_e2 = predict\n\n"
    extra = (ROOT / "variants/e3_same_item.py").read_text()
    extra = extra.replace("from submission_e2.model import predict as predict_e2, e1b, clip, sigmoid", "")
    # Keep the configurable implementation, then expose the fixed submission API.
    source += extra + "\n\nE3_CONFIG = " + repr(config)
    source += "\nE3_IMPLEMENTATION = predict\n\ndef predict(input, labeled=None):\n    return E3_IMPLEMENTATION(input, labeled, **E3_CONFIG)\n"
    (folder / "model.py").write_text(source)
    inherited = json.loads((ROOT / "experiments/e2_item/results.json").read_text())["assignments"]["tune"]
    (folder / "README.md").write_text(
        "# E3: cross-subject same-item residuals on frozen E2\n\n"
        "E2 is a hierarchical E1b probability plus a centered, ridge-fitted item-feature\n"
        "logit offset. Its parameters and weights are embedded unchanged in model.py.\n"
        "E1b pools acquired labels for the same anonymous benchmark, using prior mean\n"
        "0.597568381426247 and benchmark/pair shrinkage strengths of 4. E2 uses gamma=1\n"
        "and ridge penalty=10. Its 14 bounded string/metadata features describe length,\n"
        "line count, digits, non-ASCII-letter fraction, compression, code/math/choice\n"
        "cues, URLs, interactors, feature-key count, year gap, and missing-year flags.\n\n"
        "E3 matches all four item fields exactly: item_content, item_features,\n"
        "interactors, and benchmark_id. Only entries whose subject dictionary differs\n"
        "from the target subject contribute a residual. No anonymous identity is\n"
        "reverse engineered. For each match (t,y), q_t is E1b for subject t on this\n"
        "benchmark, recomputed from the entire supplied labeled list. The matching\n"
        "label stays in q_t (in-sample residual shrinkage); target-subject labels may\n"
        "influence q_t's E1b benchmark prior, but never contribute a residual.\n\n"
        "Probability mode: clip(p_E2 + eta * sum(y-q_t)/(n+lambda)).\n"
        "Logit mode: clip(sigmoid(logit(p_E2) + eta * sum((y-q_t)/\n"
        "max(q_t*(1-q_t),0.05))/(n+lambda))). The variance floor bounds extreme\n"
        "approximate logit residuals. All matching acquired entries are counted.\n"
        f"Selected configuration: lambda={config['shrinkage']}, eta={config['eta']}, mode={config['mode']}.\n\n"
        "The grid was lambda in {0.5,1,2,4,8}, eta in {0.5,1}, and both modes;\n"
        "selection minimized balanced Brier ALC on tuning seeds 0..19. The split\n"
        "uses SHA256 benchmark hashing with holdout_fraction=0.3. Confirmation\n"
        "seeds 1000..1049 and packed sensitivity were not used for selection.\n\n"
        "Training data: measurement-db revision 54074453 (requester-specified;\n"
        "no network verification), binary item benchmarks, subjects released in\n"
        "2025-2026, and eligible subject-benchmark pairs with at least 80 items.\n"
        "Only the tuning benchmarks below were used for E2 training and E3 tuning.\n\n" +
        "\n".join("- " + b for b in sorted(set(inherited) | set(benchmarks))) +
        "\n\nOnly the Python standard library is required. No persistent call state, I/O,\n"
        "or custom acquisition function is used. Outputs are clipped to [0.001,0.999].\n"
        "No matching other-subject label returns E2 exactly; malformed evidence is\n"
        "skipped and correction failures return E2. Item text is never logged.\n\n"
        "Disclosure: Anthropic Claude and OpenAI Codex provided LLM-based coding\n"
        "assistance for implementation, experiment orchestration, and documentation.\n\n"
        "Limitation: availability of same-item labels in the official evaluation\n"
        "stream is not confirmed. The local simulator is not an official-equivalence\n"
        "claim, and sparse overlap can make E3 behave almost exactly like E2.\n")


def summarize_full(result):
    runs = []
    for run in result["runs"]:
        decomposition = {str(b): {c: statistics.mean(p["checkpoints"][str(b)]["decomposition"][c]
                        for p in run["pairs"]) for c in COMPONENTS} for b in sim.BUDGETS}
        weighted = {c: math.fsum(w * decomposition[str(b)][c] for b, w in zip(sim.BUDGETS, sim.WEIGHTS))
                    for c in COMPONENTS}
        runs.append({"seed": run["seed"], "alc": run["alc"], "brier_by_budget": run["brier_by_budget"],
                     "decomposition_by_budget": decomposition, "decomposition_alc": weighted,
                     "pairs": len(run["pairs"]),
                     "evaluation_prediction_calls_by_budget": run["evaluation_prediction_calls_by_budget"]})
    return {"summary": result["summary"], "conditions": result["conditions"], "runs": runs,
            "eligible_pair_count": result["eligible_pair_count"],
            "decomposition_alc": {c: statistics.mean(r["decomposition_alc"][c] for r in runs) for c in COMPONENTS},
            "decomposition_by_budget": {str(b): {c: statistics.mean(r["decomposition_by_budget"][str(b)][c]
                for r in runs) for c in COMPONENTS} for b in sim.BUDGETS}}


def measure(data, name, strategy="balanced", runs=50, seed=1000, split="holdout"):
    started = time.monotonic()
    module = load_model(ROOT / f"submission_{name.lower()}/model.py")
    times, coverage = [], {str(b): {"covered": 0, "predictions": 0} for b in sim.BUDGETS}
    last_n, budget = -1, 0

    def predict(x, labeled):
        nonlocal last_n, budget
        if name == "E3":
            if len(labeled) != last_n:
                counts = Counter(sim.canonical([s, item.get("benchmark_id")]) for (s, item), _ in labeled)
                budget = max(counts.values(), default=0)
                last_n = len(labeled)
            row = coverage[str(budget)]
            row["predictions"] += 1
            row["covered"] += int(any(s != x[0] and all(k in item and k in x[1] and item[k] == x[1][k]
                for k in e3.ITEM_FIELDS) for (s, item), _ in labeled))
        t = time.perf_counter()
        p = module.predict(x, labeled)
        times.append(time.perf_counter() - t)
        if len(times) % 15000 == 0:
            print(f"{name} {strategy}: {len(times)} predictions, {time.monotonic()-started:.1f}s, RSS {rss_mib():.1f} MiB", flush=True)
        return p

    full = sim.run_simulation(data, predict, split=split, holdout_fraction=0.3,
                             pairs_strategy=strategy, runs=runs, seed=seed)
    result = summarize_full(full)
    result["prediction_timing"] = {"calls": len(times), "median_seconds": statistics.median(times),
        "maximum_seconds": max(times), "mean_seconds": statistics.mean(times),
        "scope": "real full-content model.predict; excludes simulator deepcopy and coverage instrumentation"}
    if name == "E3":
        for row in coverage.values():
            row["fraction"] = row["covered"] / row["predictions"] if row["predictions"] else 0.0
        result["same_item_coverage"] = coverage
        for b in sim.BUDGETS:
            assert coverage[str(b)]["predictions"] == sum(r["evaluation_prediction_calls_by_budget"][str(b)] for r in full["runs"])
    result["seconds"] = time.monotonic() - started
    result["max_rss_mib"] = rss_mib()
    result["model_sha256"] = hashlib.sha256((ROOT / f"submission_{name.lower()}/model.py").read_bytes()).hexdigest()
    path = OUT / f"{split}_{strategy}_{name.lower()}.json"
    dump(path, result)
    print(f"Completed {name} {strategy}: ALC {result['summary']['alc_mean']:.9f}, {result['seconds']:.1f}s", flush=True)
    return result


def parity(data, tuning):
    checks = []
    for name in ("E3", "E2"):
        actual = measure(data, name, runs=1, seed=0, split="tune")["runs"][0]
        if name == "E2":
            replay = tuning["E2"]["runs"][0]
        else:
            replay = next(row for row in tuning["grid"] if row["config"] == tuning["config"])["runs"][0]
        error = max(abs(actual["brier_by_budget"][str(b)] - replay["brier_by_budget"][str(b)]) for b in sim.BUDGETS)
        assert error < 1e-12
        checks.append({"method": name, "seed": 0, "max_budget_brier_difference": error})
    dump(OUT / "parity.json", checks)
    return checks


def synthetic_latency(model):
    subject = {"normalized_name": "target", "release_date": "2025-01-01"}
    item = {"item_content": "Synthetic task 2024. def f(): return 1;\n" * 25000,
            "item_features": "{}", "interactors": "", "benchmark_id": "synthetic"}
    rows = []
    for n, matching in ((0, False), (31, False), (372, False), (372, True)):
        labels = [[[dict(subject, normalized_name=f"other-{j//31}"),
                    item if matching else dict(item, item_content=f"Synthetic observed {j}. " * 220)], j % 2]
                  for j in range(n)]
        times = []
        for _ in range(25):
            t = time.perf_counter()
            p = model.predict([subject, item], labels)
            times.append(time.perf_counter()-t)
            assert .001 <= p <= .999
        rows.append({"labeled_entries": n, "all_same_item": matching, "body_chars": len(item["item_content"]),
                     "calls": len(times), "median_seconds": statistics.median(times), "maximum_seconds": max(times)})
    return rows


def finalize():
    tuning = json.loads((OUT / "tune.json").read_text())
    holdout = {name: json.loads((OUT / f"holdout_balanced_{name.lower()}.json").read_text())
               for name in ("E3", "E2", "E1b")}
    packed = {name: json.loads((OUT / f"holdout_packed_{name.lower()}.json").read_text()) for name in ("E3", "E2")}
    for ref in ("E2", "E1b"):
        holdout[f"paired_e3_minus_{ref.lower()}"] = paired(holdout["E3"]["runs"], holdout[ref]["runs"])
    packed["paired_e3_minus_e2"] = paired(packed["E3"]["runs"], packed["E2"]["runs"])
    result = {"data_revision": "54074453",
              "config": tuning["config"], "tuning": tuning, "holdout": holdout, "packed": packed,
              "parity_checks": json.loads((OUT / "parity.json").read_text()),
              "synthetic_latency": synthetic_latency(load_model(ROOT / "submission_e3/model.py")),
              "resources": json.loads((OUT / "resources.json").read_text()),
              "limitations": ["本番で同じ問題の他 subject ラベルが供給されるかは未確認です。",
                 "これはローカル模擬装置での検証であり、公式評価との等価性は保証していません。",
                 "ベンチマーク単位の調整・確認分離は保ちますが、E1b/E2 と同じ確認用集合を再使用しています。",
                 "照合された他 subject の該当ラベル自体を q_t に含めるため、残差は学習内で縮みます。",
                 "seed 間で同じデータを再利用するため、標準誤差は新しいベンチマークへの一般化を保証しません。"]}
    d = holdout["paired_e3_minus_e2"]
    result["recommend_submission"] = d["mean"] < -2*d["standard_error"] and packed["paired_e3_minus_e2"]["mean"] <= 0
    result["recommendation"] = ("E3 を提出候補として推奨します。" if result["recommend_submission"] else "今回の結果から E3 への置換は推奨しません。")
    if (OUT / "verification.json").exists():
        result["verification"] = json.loads((OUT / "verification.json").read_text())
    dump(OUT / "results.json", result)
    (OUT / "summary.md").write_text(summary(result))
    print(result["recommendation"], flush=True)
    return result


def summary(r):
    h, p = r["holdout"], r["packed"]
    lines = ["# E3 同じ問題の他 subject ラベルによる補正", "", r["recommendation"], "",
        "Brier ALC は予算0/1/3/7/15/31の Brier を0.1/0.2/0.2/0.2/0.2/0.1で平均し、subject–benchmark の組を等しく平均した損失です。小さいほど良い値です。標準偏差は50 seedsの母標準偏差、対の差の標準誤差は差の標本標準偏差÷√50です。", "",
        f"選択: λ={r['config']['shrinkage']}、η={r['config']['eta']}、足し方={r['config']['mode']}。調整用のみ、holdout_fraction=0.3、balanced、seeds 0〜19で20通りを比較しました。確認用での再選択はしていません。", "",
        "確率加算は p=clip(p_E2+ηΣ(y−q_t)/(n+λ)) です。ロジット加算は logit(p)=logit(p_E2)+ηΣ((y−q_t)/max(q_t(1−q_t),0.05))/(n+λ) です。q_t は供給された labeled 全体から計算する E1b で、該当ラベル自体を含みます。対象 subject のラベルは残差の項として数えませんが、E2 と q_t の事前確率には通常の E1b と同様に入ります。", "",
        "4項目の完全一致で同じ問題を判定します。本文は比較と特徴計算だけに使い、保存・表示しません。匿名 ID の正体は推定しません。失敗・一致なしでは E2 の値を返します。", "",
        "## 確認用 balanced（seeds 1000〜1049）", "",
        "元の run_simulation を各手法で直接実行しました。本文の空文字への置換や模擬装置の変更はありません。", "",
        "| 手法 | ALC 平均 ± 標準偏差 |", "|---|---:|"]
    for name in ("E3", "E2", "E1b"):
        s = h[name]["summary"]
        lines.append(f"| {name} | {s['alc_mean']:.9f} ± {s['alc_std']:.9f} |")
    lines += ["", "| 対の差 | 平均 ± 標準誤差 | E3 が良い seed | 同値 |", "|---|---:|---:|---:|"]
    for name in ("E2", "E1b"):
        d = h[f"paired_e3_minus_{name.lower()}"]
        lines.append(f"| E3−{name} | {d['mean']:+.9f} ± {d['standard_error']:.9f} | {d['better_seeds']}/50 | {d['equal_seeds']}/50 |")
    lines += ["", "| 手法 | 予算0 | 予算1 | 予算3 | 予算7 | 予算15 | 予算31 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("E3", "E2", "E1b"):
        lines.append("| " + name + " | " + " | ".join(f"{h[name]['summary']['brier_by_budget'][str(b)]['mean']:.9f}" for b in sim.BUDGETS) + " |")
    lines += ["", "## 同じ問題の他 subject ラベルが1件以上あった割合", "",
        "分母は模擬装置が各予算で実際に呼び出した評価予測です。同じ入力の予算内キャッシュにより重複は1回です。全50 seedsを合算した予測件数で割合を計算します。", "",
        "| 予算 | 該当予測数 / 全評価予測数 | 割合 |", "|---:|---:|---:|"]
    for b in sim.BUDGETS:
        c = h["E3"]["same_item_coverage"][str(b)]
        lines.append(f"| {b} | {c['covered']} / {c['predictions']} | {100*c['fraction']:.4f}% |")
    lines += ["", "## Brier の3成分", "",
        "各組・予算で tools/decompose.py と同じ分解を計算し、組と seeds を等しく平均しました。水準のずれは平均予測と平均実測の差の二乗、形のずれは問題ごとの中心化予測と中心化実測の差の二乗、実測の揺らぎは同じ入力内のラベルの分散です。下表は ALC と同じ予算重みを付けた値です。", "",
        "| 手法 | 水準のずれ | 形のずれ | 実測の揺らぎ |", "|---|---:|---:|---:|"]
    for name in ("E3", "E2", "E1b"):
        lines.append("| " + name + " | " + " | ".join(f"{h[name]['decomposition_alc'][c]:.9f}" for c in COMPONENTS) + " |")
    reduction = h["E2"]["decomposition_alc"]["shape_mismatch"] - h["E3"]["decomposition_alc"]["shape_mismatch"]
    lines += ["", f"E2 に対する形のずれの削減: {reduction:+.9f}（{100*reduction/h['E2']['decomposition_alc']['shape_mismatch']:+.4f}%）。", "",
        "| 予算 | E3 水準 | E3 形 | E3 揺らぎ | E2 形−E3 形 |", "|---:|---:|---:|---:|---:|"]
    for b in sim.BUDGETS:
        c = h["E3"]["decomposition_by_budget"][str(b)]
        red = h["E2"]["decomposition_by_budget"][str(b)]["shape_mismatch"] - c["shape_mismatch"]
        lines.append(f"| {b} | {c['level_shift']:.9f} | {c['shape_mismatch']:.9f} | {c['empirical_fluctuation']:.9f} | {red:+.9f} |")
    lines += ["", "## packed の感度（同じ確認用50 seeds）", "", "| 手法 | ALC 平均 ± 標準偏差 |", "|---|---:|"]
    for name in ("E3", "E2"):
        s = p[name]["summary"]
        lines.append(f"| {name} | {s['alc_mean']:.9f} ± {s['alc_std']:.9f} |")
    d = p["paired_e3_minus_e2"]
    lines += ["", f"E3−E2: {d['mean']:+.9f} ± {d['standard_error']:.9f}、良い {d['better_seeds']}/50、同値 {d['equal_seeds']}/50。", "",
        "| 予算 | E3 Brier | E2 Brier | 同じ問題ラベルの割合 |", "|---:|---:|---:|---:|"]
    for b in sim.BUDGETS:
        lines.append(f"| {b} | {p['E3']['summary']['brier_by_budget'][str(b)]['mean']:.9f} | {p['E2']['summary']['brier_by_budget'][str(b)]['mean']:.9f} | {100*p['E3']['same_item_coverage'][str(b)]['fraction']:.4f}% |")
    lines += ["", "## 調整用の全候補", "", "| 足し方 | λ | η | ALC |", "|---|---:|---:|---:|"]
    for row in r["tuning"]["grid"]:
        c = row["config"]
        lines.append(f"| {c['mode']} | {c['shrinkage']} | {c['eta']} | {row['summary']['alc_mean']:.9f} |")
    lines += ["", "調整用ベンチマーク: " + ", ".join(r["tuning"]["benchmarks"]) + "。", "",
        "データは measurement-db revision 54074453（依頼者指定）です。E2 の重みを固定し、E3 の定数だけを調整しました。学習データと Anthropic Claude / OpenAI Codex のコーディング支援を英語 README に開示しました。", "",
        "## 時間・メモリ・検証", "", f"実験命令: `{r['resources']['command']}`。実験の経過時間 {r['resources']['elapsed_seconds']:.3f} 秒、最大常駐メモリ {r['resources']['max_rss_mib']:.3f} MiB。同時実験プロセスは最大 {r['resources'].get('parallel_experiment_processes', 1)} 本、Arrow は各1スレッドです。メモリはプロセス単位の最大値で、2本を合算したピークではありません。", "",
        f"調整の高速集計と本物の経路の照合: {r['parity_checks']}。", "",
        "| 実データ予測 | 呼出し数 | 中央値ms | 最大ms |", "|---|---:|---:|---:|"]
    for strategy, methods in (("balanced", h), ("packed", p)):
        t = methods["E3"]["prediction_timing"]
        lines.append(f"| E3 {strategy} | {t['calls']} | {1000*t['median_seconds']:.4f} | {1000*t['maximum_seconds']:.4f} |")
    lines += ["", "実データの計時は model.predict のみで、模擬装置のコピーと割合計測を除きます。", "",
        "| 合成入力のラベル数 | 全ラベルが同じ問題 | 中央値ms | 最大ms |", "|---:|---|---:|---:|"]
    for t in r["synthetic_latency"]:
        lines.append(f"| {t['labeled_entries']} | {t['all_same_item']} | {1000*t['median_seconds']:.4f} | {1000*t['maximum_seconds']:.4f} |")
    if "verification" in r:
        v = r["verification"]
        tests = v.get("tests", {})
        lines += ["", f"試験: {tests.get('tests_run', 0)} 件、失敗 {tests.get('failures', 0)} 件、エラー {tests.get('errors', 0)} 件。", "",
                  f"試験命令: `{tests.get('command', '')}`。unittest discover と同じ全試験を収集し、既存試験の一時保存先だけを許可範囲へ振り替えました。", "",
                  f"zip 作成: `{v.get('build_command', '')}`。", "",
                  f"zip 検査: `{v.get('zip_check_command', '')}`。結果: {v.get('zip_check', '')}。", "",
                  f"全体の経過時間: {v.get('total_wall_seconds', 0):.3f} 秒。最大観測プロセスメモリ: {v.get('maximum_observed_process_rss_mib', 0):.3f} MiB。"]
        if "parallel_command" in r["resources"]:
            lines += ["", f"packed の並行測定: `{r['resources']['parallel_command']}`。", "",
                      "主プロセスは balanced の3手法完了後に、別プロセスと重複する packed 工程へ入った所で停止しました。packed の集計には別プロセスで完了した50 seedsだけを採用しています。"]
    lines += ["", "## 限界", ""] + ["- " + x for x in r["limitations"]]
    return "\n".join(lines) + "\n"


def run_tests():
    """Run unchanged discovery, relocating only existing fixture write paths."""
    import tempfile
    import unittest
    from unittest.mock import patch
    from tools import tune_e1
    original = tempfile.TemporaryDirectory
    def temporary(*args, **kwargs):
        if Path(kwargs.get("dir") or "/") in (ROOT, ROOT / "experiments/e1_tuning"):
            kwargs["dir"] = OUT
        return original(*args, **kwargs)
    with patch.object(tempfile, "TemporaryDirectory", temporary), patch.object(tune_e1, "OUT", OUT):
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py"))
    dump(OUT / "tests.json", {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "successful": result.wasSuccessful(), "max_rss_mib": rss_mib(),
        "command": "PYTHONPATH=. .venv/bin/python -B tools/train_e3.py --test-suite",
        "discovery": "unittest discover -s tests -p 'test_*.py'",
        "fixture_redirect": "only existing ROOT/e1_tuning temporary paths -> experiments/e3_item; tests unchanged"})
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--test-suite", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    if args.test_suite:
        return run_tests()
    if args.finalize:
        finalize()
        return 0
    started = time.monotonic()
    tuning = tune(args.data)
    emit_submission(tuning["config"], tuning["benchmarks"])
    parity(args.data, tuning)
    for name in ("E3", "E2", "E1b"):
        measure(args.data, name)
    for name in ("E3", "E2"):
        measure(args.data, name, "packed")
    dump(OUT / "resources.json", {"elapsed_seconds": time.monotonic()-started, "max_rss_mib": rss_mib(),
        "command": "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -B tools/train_e3.py",
        "parallel_experiment_processes": 1, "network_used": False})
    finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
