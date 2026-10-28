"""Offline, single-process E1 tuning. Never reads the items.content column."""

import argparse
from contextlib import contextmanager
from functools import partial
import hashlib
import itertools
import json
import math
from pathlib import Path
import resource
import statistics
import sys
import time
from unittest.mock import patch

sys.dont_write_bytecode = True

import pyarrow as pa
import pyarrow.parquet as pq
from f1 import simulate as sim
from f1.splits import benchmark_split
from variants import from_counts, identify, predict, summarize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/e1_tuning"
YEARS = {"2025", "2026"}
ALLOW = {
    "benchmarks": {"benchmark_id", "response_type", "granularity", "release"},
    "subjects": {"subject_id", *sim.SUBJECT_FIELDS},
    "items": {"item_id", "benchmark_id", "item_features"},
    "responses": set(sim.RESPONSE_COLUMNS),
}


def safe_rows(folder, table, columns):
    """Project before decoding; content is never passed to Arrow, even in tests."""
    requested = set(columns)
    actual = [c for c in columns if not (table == "items" and c == "content")]
    if table not in ALLOW or not set(actual) <= ALLOW[table]:
        raise ValueError("Unexpected table/column request")
    for batch in pq.ParquetFile(Path(folder) / (table + ".parquet")).iter_batches(
            batch_size=8192, columns=actual, use_threads=False):
        for row in batch.to_pylist():
            if table == "items" and "content" in requested:
                row["content"] = ""
            yield row


@contextmanager
def content_free_f1():
    with patch.object(sim, "read_rows", safe_rows):
        yield


def public_benchmarks(data):
    rows = []
    for folder in sorted(data.glob("*/formatted_tables")):
        for row in safe_rows(folder, "benchmarks", ("benchmark_id", "response_type", "granularity", "release")):
            row = dict(row, dataset=folder.parent.name)
            row["selected"] = (row["response_type"] == "binary" and
                               row["granularity"] == "item" and row["release"] == "public")
            row["split"] = benchmark_split(row["benchmark_id"], 0.3)
            rows.append(row)
    return rows


def tune_prior(pairs):
    totals = {}
    wanted = {p.key for p in pairs}
    for folder in sorted({p.folder for p in pairs}):
        for row in safe_rows(folder, "responses", ("subject_id", "benchmark_id", "response")):
            key = row["benchmark_id"], row["subject_id"]
            if key not in wanted:
                continue
            y = sim.response_label(row)
            if y is not None:
                s, n = totals.get(key, (0, 0))
                totals[key] = s + y, n + 1
    rates = [{"benchmark_id": b, "subject_id": s, "successes": v[0], "response_rows": v[1],
              "rate": v[0] / v[1]} for (b, s), v in sorted(totals.items())]
    return statistics.mean(r["rate"] for r in rates), rates


def make_plan(pairs, seed, strategy, *, max_sample=1000):
    """Replay exactly F1's state preparation and prediction-independent acquisition.

    All predictors here are constant within a subject-benchmark pair. Thus the
    response-weighted Brier equals p*p - 2*p*ybar + ybar, without reading text.
    """
    allocations = sim.allocate_pairs(pairs, seed, strategy, max_sample)
    states = sim.prepare_states(allocations, seed, YEARS)
    labeled, checkpoints = [], []
    audit = {"seed": seed, "strategy": strategy, "pairs": [],
             "v3_comparator_pairs_by_budget": {}}
    digest = hashlib.sha256()
    for state in states:
        audit["pairs"].append({"benchmark_id": state.pair.benchmark,
                               "subject_id": state.pair.subject, "sampled_items": state.n_sampled})
        digest.update(sim.canonical([state.pair.key, state.evaluation_ids]).encode())
    for budget in sim.BUDGETS:
        for state in states:
            while len(state.acquired_ids) < budget and state.cursor < len(state.acquisition):
                iid, input, label = state.acquisition[state.cursor]
                acquired = len(state.acquired_ids)
                context = {"subject_id": sim.anonymous("subject", state.pair.subject),
                           "benchmark_id": sim.anonymous("benchmark", state.pair.benchmark),
                           "labels_acquired": acquired, "labels_remaining": sim.MAX_LABELS - acquired,
                           "max_labels": sim.MAX_LABELS,
                           "items_remaining": len(state.acquisition) - state.cursor}
                state.cursor += 1
                if sim.default_acquire(input, context):
                    state.acquired_ids.append(iid)
                    labeled.append([input, label])
        counts = summarize(labeled)
        evaluation = []
        for state in states:
            successes = sum(sum(ys) for _, _, ys in state.evaluation)
            n = sum(len(ys) for _, _, ys in state.evaluation)
            sid, bid = identify(state.evaluation[0][1])
            evaluation.append((sid, bid, successes / n))
            digest.update(sim.canonical([budget, state.pair.key, state.acquired_ids,
                                         successes, n]).encode())
        checkpoints.append((counts, evaluation))
        usable = 0
        for sid, bid, _ in evaluation:
            if any(own_sid == sid and other_bid != bid and own_n > 0 and
                   any(peer_bid == other_bid and peer_sid != sid and peer_n > 0
                       for (peer_sid, peer_bid), (_, peer_n) in counts.items())
                   for (own_sid, other_bid), (_, own_n) in counts.items()):
                usable += 1
        audit["v3_comparator_pairs_by_budget"][str(budget)] = usable
    audit["trajectory_sha256"] = digest.hexdigest()
    audit["labels_acquired_at_31"] = [len(s.acquired_ids) for s in states]
    return {"seed": seed, "checkpoints": checkpoints, "audit": audit}


def score_plan(plan, config):
    scores = {}
    for budget, (counts, evaluation) in zip(sim.BUDGETS, plan["checkpoints"]):
        cells = []
        for subject, benchmark, mean_y in evaluation:
            p = from_counts(subject, benchmark, counts, **config)
            cells.append(p * p - 2 * p * mean_y + mean_y)
        scores[str(budget)] = statistics.mean(cells)
    return {"seed": plan["seed"], "alc": math.fsum(w * scores[str(b)] for b, w in
            zip(sim.BUDGETS, sim.WEIGHTS)), "brier_by_budget": scores}


def describe(runs):
    return {"alc_mean": statistics.mean(r["alc"] for r in runs),
            "alc_std": statistics.pstdev(r["alc"] for r in runs),
            "brier_by_budget": {str(b): {"mean": statistics.mean(r["brier_by_budget"][str(b)] for r in runs),
                "std": statistics.pstdev(r["brier_by_budget"][str(b)] for r in runs)} for b in sim.BUDGETS}}


def paired(runs, baseline):
    assert [r["seed"] for r in runs] == [r["seed"] for r in baseline]
    differences = [r["alc"] - b["alc"] for r, b in zip(runs, baseline)]
    return {"mean": statistics.mean(differences),
            "standard_error": statistics.stdev(differences) / math.sqrt(len(differences))
            if len(differences) > 1 else 0.0, "n": len(differences)}


def parity_check(data, benchmarks, pairs):
    """Independent scoring path: actual run_simulation vs aggregated replay."""
    checks = []
    for strategy in ("balanced", "packed"):
        plan = make_plan(pairs, 0, strategy)
        for variant in ("constant", "empirical", "V1", "V2", "V3"):
            config = dict(variant=variant, k=4, b=2, a=8, prior_mean=0.43)
            reference = sim.run_simulation(data, partial(predict, **config), benchmarks=benchmarks,
                runs=1, seed=0, split="tune", holdout_fraction=0.3, pairs_strategy=strategy)
            fast = score_plan(plan, config)
            error = max(abs(fast["brier_by_budget"][str(b)] -
                            reference["runs"][0]["brier_by_budget"][str(b)]) for b in sim.BUDGETS)
            if error > 1e-12:
                raise ArithmeticError("Replay differs from F1")
            checks.append({"variant": variant, "strategy": strategy, "maximum_brier_error": error})
    return checks


def suite():
    # Existing F1 tests request a temporary directory directly below code/.
    # Redirect only that request to the user's explicitly writable output tree.
    import tempfile
    import unittest
    OUT.mkdir(parents=True, exist_ok=True)
    original = tempfile.TemporaryDirectory

    def temporary(*args, **kwargs):
        if Path(kwargs.get("dir") or "/") == ROOT:
            kwargs["dir"] = OUT
        return original(*args, **kwargs)

    started = time.monotonic()
    with patch.object(tempfile, "TemporaryDirectory", temporary):
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py"))
    payload = {"tests_run": result.testsRun, "failures": len(result.failures),
               "errors": len(result.errors), "successful": result.wasSuccessful(),
               "elapsed_seconds": time.monotonic() - started, "max_rss_mib": max_rss()}
    (OUT / "tests.json").write_text(json.dumps(payload, indent=2) + "\n")
    return 0 if result.wasSuccessful() else 1


def max_rss():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024 if sys.platform == "darwin" else 1024)


def summary(result):
    best, hold = result["best"], result["holdout"]["balanced"]
    lines = ["# E1 の定数調整と確認実験", "", "本文を使わないローカル模擬実験です。ALC と Brier は小さいほど良い値です。",
        "ALC は各組の予算別 Brier を 0.1/0.2/0.2/0.2/0.2/0.1 で平均し、組を等しく平均した値です。",
        "標準偏差は seed ごとの組平均の母標準偏差です。対の差は手法−比較相手です。標準誤差は差の標本標準偏差÷√50です。", "",
        f"調整用の組別正答率を等しく平均した PRIOR_MEAN は **{result['prior']['calculated']:.12f}** です。",
        "組内では全有効応答行を等しく数えます。確認用の応答は PRIOR_MEAN の推定に含めません。", "",
        "## 調整用の最良", "", "| 手法 | PRIOR_MEAN | k | b | a | 平均 ALC |", "|---|---:|---:|---:|---:|---:|"]
    for v, entry in best.items():
        c = entry["config"]
        lines.append(f"| {v} | {c['prior_mean']:.9f} | {c['k']} | {c.get('b', '—')} | {c.get('a', '—')} | {entry['summary']['alc_mean']:.8f} |")
    lines += ["", "## 確認用の比較（balanced、50 seeds）", "",
        "| 手法 | 平均 ALC ± 標準偏差 | 対 0.5 の差 ± 標準誤差 | 対 Empirical Mean の差 ± 標準誤差 |", "|---|---:|---:|---:|"]
    for name, entry in hold.items():
        s, d = entry["summary"], entry["paired_differences"]
        lines.append(f"| {name} | {s['alc_mean']:.8f} ± {s['alc_std']:.8f} | {d['constant']['mean']:+.8f} ± {d['constant']['standard_error']:.8f} | {d['empirical']['mean']:+.8f} ± {d['empirical']['standard_error']:.8f} |")
    lines += ["", "## 予算別 Brier（確認用 balanced、seed 平均）", "", "| 手法 | 0 | 1 | 3 | 7 | 15 | 31 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, entry in hold.items():
        lines.append("| " + name + " | " + " | ".join(f"{entry['summary']['brier_by_budget'][str(b)]['mean']:.8f}" for b in sim.BUDGETS) + " |")
    lines += ["", "## 感度（同じ最良定数、50 seeds）", "", "| 手法 | balanced 平均 ± 標準偏差 | packed 平均 ± 標準偏差 | packed 対 0.5 | packed 対 Empirical Mean |", "|---|---:|---:|---:|---:|"]
    for name in ("V1", "V2", "V3"):
        p = result["holdout"]["packed"][name]
        s, d = p["summary"], p["paired_differences"]
        lines.append(f"| {name} | {hold[name]['summary']['alc_mean']:.8f} ± {hold[name]['summary']['alc_std']:.8f} | {s['alc_mean']:.8f} ± {s['alc_std']:.8f} | {d['constant']['mean']:+.8f} ± {d['constant']['standard_error']:.8f} | {d['empirical']['mean']:+.8f} ± {d['empirical']['standard_error']:.8f} |")
    lines += ["", "## 推奨と限界", "", result["recommendation"], ""]
    lines += [f"- {item}" for item in result["limitations"]]
    coverage = []
    for key, audits in result["trajectories"].items():
        active = sum(sum(x["v3_comparator_pairs_by_budget"].values()) for x in audits)
        total = sum(len(x["pairs"]) * len(sim.BUDGETS) for x in audits)
        coverage.append(f"{key} は {active}/{total} 件")
    lines += ["- V3 の比較情報が存在した組・予算・seed の組み合わせは、" + "、".join(coverage) +
              "でした。分母には予算 0 も含みます。packed では補正が働かず V2 と V3 が一致します。"]
    lines += ["", "## 条件・実行記録", "", f"- データ revision: {result['data_revision']}（依頼者指定値）。",
        f"- 適格ベンチマーク: {result['conditions']['eligible_benchmarks']} 本。調整用 {len(result['assignments']['tune'])} 本、確認用 {len(result['assignments']['holdout'])} 本。",
        "- 調整用 seed 0〜19、確認用 seed 1000〜1049。holdout_fraction=0.3、公開年 2025〜2026、各組 80 問以上、1 seed の標本上限 1,000 subject–item。",
        f"- 格子: {len(result['grid'])} 候補。縮小記録: {json.dumps(result['grid_reductions'], ensure_ascii=False)}。",
        f"- 元の F1 採点経路との照合は 5 手法×2 配分方式で実施しました。予算別 Brier の最大絶対差は {max(c['maximum_brier_error'] for c in result['parity_checks']):.3g} です。",
        f"- 単一プロセス・Arrow スレッド数 1。経過 {result['resources']['elapsed_seconds']:.3f} 秒、最大常駐メモリ {result['resources']['max_rss_mib']:.3f} MiB。",
        "- 実行命令: `PYTHONPATH=. .venv/bin/python -B tools/tune_e1.py`。",
        "- 全格子・seed 別 ALC と Brier・分割・獲得過程の識別ハッシュ・F1 との照合は results.json にあります。", ""]
    if (OUT / "time.log").exists() and "Operation not permitted" in (OUT / "time.log").read_text():
        lines += ["- 初回の外部タイマー time -l は macOS の sysctl 制限で終了時に失敗しました。上記は外部タイマーを外した正常終了の再実行で、Python の monotonic と getrusage による計測値です。", ""]
    if (OUT / "tests.json").exists():
        tests = json.loads((OUT / "tests.json").read_text())
        lines += [f"- 試験: {tests['tests_run']} 件、失敗 {tests['failures']} 件、エラー {tests['errors']} 件。経過 {tests['elapsed_seconds']:.3f} 秒、最大常駐メモリ {tests['max_rss_mib']:.3f} MiB。",
                  "- 試験命令: `PYTHONPATH=. .venv/bin/python -B tools/tune_e1.py --test-suite`。既存試験の一時保存先だけを許可範囲へ移し、unittest discover -s tests -p 'test_*.py' と同じ試験集合を実行します。", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT.parent / "_data/mdb")
    parser.add_argument("--test-suite", action="store_true")
    args = parser.parse_args(argv)
    if args.test_suite:
        return suite()
    started = time.monotonic()
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    OUT.mkdir(parents=True, exist_ok=True)
    phases = {}
    with content_free_f1():
        metadata = public_benchmarks(args.data)
        benchmarks = {r["benchmark_id"] for r in metadata if r["selected"]}
        all_pairs, inventory = sim.inventory(args.data, benchmarks, "all", 0.3, YEARS)
        pairs = {s: [p for p in all_pairs if benchmark_split(p.benchmark, 0.3) == s]
                 for s in ("tune", "holdout")}
        prior, prior_rates = tune_prior(pairs["tune"])
        print(f"Inventory: {len(all_pairs)} eligible pairs; PRIOR_MEAN={prior:.12f}", flush=True)
        t = time.monotonic()
        # Compare against F1 once on each strategy and each predictor before reuse.
        parity = parity_check(args.data, benchmarks, pairs["tune"])
        phases["parity_seconds"] = time.monotonic() - t
        plans = {}
        for split, strategy, seeds in (("tune", "balanced", range(20)),
                                      ("holdout", "balanced", range(1000, 1050)),
                                      ("holdout", "packed", range(1000, 1050))):
            t = time.monotonic()
            group = []
            for seed in seeds:
                group.append(make_plan(pairs[split], seed, strategy))
                if len(group) % 10 == 0:
                    print(f"Prepared {split}/{strategy}: {len(group)}/{len(seeds)}; elapsed={time.monotonic()-started:.1f}s", flush=True)
            plans[split, strategy] = group
            phases[f"prepare_{split}_{strategy}_seconds"] = time.monotonic() - t
        configs = []
        for variant in ("V1", "V2", "V3"):
            for m, k, b, a in itertools.product((0.5, prior), (1, 2, 4, 8, 16),
                    (1, 2, 4, 8, 16) if variant != "V1" else (None,),
                    (2, 8, 32) if variant == "V3" else (None,)):
                c = dict(variant=variant, prior_mean=m, k=k)
                if b is not None:
                    c["b"] = b
                if a is not None:
                    c["a"] = a
                configs.append(c)
        grid, reductions = [], []
        t = time.monotonic()
        for i, c in enumerate(configs):
            # Preserve both prior candidates and all seeds. If slow, keep k/b in
            # {1,4,16}; records explicitly distinguish skipped configurations.
            if time.monotonic() - started > 1200 and (c["k"] in (2, 8) or c.get("b") in (2, 8)):
                reductions.append({"config": c, "reason": "20-minute elapsed guard; reserve holdout/output time"})
                continue
            runs = [score_plan(p, c) for p in plans["tune", "balanced"]]
            grid.append({"config": c, "summary": describe(runs), "runs": runs})
        phases["grid_seconds"] = time.monotonic() - t
        best = {v: min((r for r in grid if r["config"]["variant"] == v),
                       key=lambda r: r["summary"]["alc_mean"]) for v in ("V1", "V2", "V3")}
        fixed = {"constant": {"variant": "constant"}, "empirical": {"variant": "empirical"},
                 **{v: r["config"] for v, r in best.items()}}
        holdout = {}
        for strategy in ("balanced", "packed"):
            evaluated = {v: {"config": c, "runs": [score_plan(p, c) for p in plans["holdout", strategy]]}
                         for v, c in fixed.items()}
            for v, r in evaluated.items():
                r["summary"] = describe(r["runs"])
                r["paired_differences"] = {base: paired(r["runs"], evaluated[base]["runs"])
                                            for base in ("constant", "empirical")}
            holdout[strategy] = evaluated
        # Pick the simplest variant whose paired tune loss against the tune
        # minimum is at most one standard error. Holdout never chooses constants.
        tune_minimum = min(best, key=lambda v: best[v]["summary"]["alc_mean"])
        tune_differences = {v: paired(r["runs"], best[tune_minimum]["runs"])
                            for v, r in best.items()}
        recommended = next(v for v in ("V1", "V2", "V3") if
                           tune_differences[v]["mean"] <= tune_differences[v]["standard_error"])
        assignments = {s: sorted({p.benchmark for p in ps}) for s, ps in pairs.items()}
        for row in metadata:
            row["eligible_pairs"] = sum(p.benchmark == row["benchmark_id"] for p in all_pairs)
        result = {"format": "x053_e1_tuning_v1",
            "data_revision": "54074453", "data_revision_source": "user supplied; no network verification",
            "conditions": {"data": str(args.data.resolve()), "holdout_fraction": 0.3,
                "release_years": sorted(YEARS), "minimum_items": 80, "max_sample": 1000,
                "public_binary_item_benchmarks": len(benchmarks),
                "eligible_benchmarks": len({p.benchmark for p in all_pairs}),
                "eligible_pairs": len(all_pairs), "budgets": sim.BUDGETS, "weights": sim.WEIGHTS,
                "main_pairs_strategy": "balanced", "sensitivity_strategies": ["balanced", "packed"],
                "tune_seeds": list(range(20)), "holdout_seeds": list(range(1000, 1050)),
                "planned_grid": {"k": [1,2,4,8,16], "b": [1,2,4,8,16], "a": [2,8,32], "prior_mean": [0.5,prior]},
                "processes": 1, "arrow_threads": 1, "content_read": False,
                "std": "population SD of run pair means, ddof=0",
                "paired_se": "sample SD of paired run differences / sqrt(50)",
                "recommendation_rule": "simplest V1/V2/V3 with paired tune loss vs tune minimum <= one paired standard error; no holdout reselection",
                "v3_offset": "sum(own_n*(logit(clip(own_s/own_n))-logit(clip(other_s/other_n))))/(sum(own_n)+a); other benchmarks with other_n>0 only",
                "v3_target": "sigmoid(logit(V2 benchmark prior)+offset), then own pair shrinkage k",
                "baseline_empirical": "own pair mean; no smoothing or clipping; no labels -> 0.5"},
            "benchmark_metadata": metadata, "inventory": inventory, "assignments": assignments,
            "prior": {"calculated": prior, "pair_rates": prior_rates},
            "grid": grid, "grid_reductions": reductions, "best": best, "holdout": holdout,
            "tune_minimum_variant": tune_minimum, "tune_differences_from_minimum": tune_differences,
            "recommended_variant": recommended,
            "parity_checks": parity, "trajectories": {f"{s}/{st}": [p["audit"] for p in pp]
                for (s, st), pp in plans.items()},
            "recommendation": (f"推奨は {recommended} です。調整用の最小値との差が、対の差の標準誤差以下である手法の中から最も単純な方式を選びました。"
                f"{recommended} と調整用最小の {tune_minimum} の差は {tune_differences[recommended]['mean']:+.8f}、標準誤差は {tune_differences[recommended]['standard_error']:.8f} です。"
                "V3 の追加補正による改善はこの実験では明確ではありません。確認用で定数を選び直していません。"),
            "limitations": [
                "items.content は列射影で除外し、F1 入力では空文字に置換しました。既定獲得は入力全体をハッシュするため、本文を含む通常の F1 と獲得対象が変わり得ます。全手法はこの同一の獲得過程を共有します。",
                "F1 のローカル近似です。本番・非公開ベンチマーク・順位への一般化は確認していません。",
                "50 seeds の標準誤差は同じ有限データ上の再標本化の揺らぎです。独立したベンチマークを追加した場合の不確実性を表しません。",
                "V3 は別ベンチマークに同じ subject 属性があり、比較する他 subject の獲得ラベルもある場合だけ働きます。極端な経験率は [0.001,0.999] に収めてロジットを計算します。",
                "Empirical Mean は指定に従い平滑化せず、0 と 1 を許します。V1/V2/V3 の出力は [0.001,0.999] です。",
                "PRIOR_MEAN は調整用の全適格組を用いた経験的な値で、20 seeds の評価標本も含みます。確認用ベンチマークのラベルは含みません。",
                "balanced は最大 12 組、packed は少数の組へ集中します。共有ラベルと V3 の補正に使える重なりが少ない条件があります。"],
            "resources": {"elapsed_seconds": time.monotonic()-started, "max_rss_mib": max_rss(), "phases": phases}}
    def write_result():
        with (OUT / "results.json").open("w") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    write_result()
    (OUT / "summary.md").write_text(summary(result))
    result["resources"].update(elapsed_seconds=time.monotonic()-started, max_rss_mib=max_rss(),
        measurement_scope="entire experiment including initial artifact writes; excludes final metadata refresh")
    write_result()
    (OUT / "summary.md").write_text(summary(result))
    print(json.dumps({"best": {v: r["config"] for v,r in best.items()},
                      "resources": result["resources"], "recommended": recommended}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
