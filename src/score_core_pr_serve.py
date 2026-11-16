#!/usr/bin/env python3
"""Score paired serving runs from the core generic-default PR.

Usage: python3 src/score_core_pr_serve.py --tags h100_q25,h100_gemma [--merge]
Raw benchmark JSON and logs must already be in results/validation/core_pr_serving/<tag>.
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "validation" / "core_pr_serving"
WORKLOAD_METRIC = {
    "decode32": "output_throughput",
    "prefill32": "request_throughput",
    "decode1": "output_throughput",
    "longdecode32": "output_throughput",
}
ARMS = ("off", "base", "pr")
ROUNDS = (1, 2, 3)


def score_tag(tag):
    directory = DATA / tag
    assert (directory / "DONE").exists(), f"incomplete job: {tag} (missing DONE)"
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["base_sha"] == "df8c0296af83ac9807a473fbb01e614a643964c0", metadata
    assert metadata["pr_sha"] == "beae62efc9506dd3df5cfe8c80c68f30ca45245f", metadata

    values = {}
    for workload, metric in WORKLOAD_METRIC.items():
        for arm in ARMS:
            for round_id in ROUNDS:
                name = f"bench_{workload}_{arm}_r{round_id}.json"
                path = directory / name
                assert path.exists(), f"missing result: {path}"
                record = json.loads(path.read_text())
                assert record.get("failed", 0) == 0, f"failed requests in {name}"
                assert record.get("completed") == record.get("num_prompts"), f"incomplete prompts in {name}"
                assert record.get("model_id") == metadata["model"], (name, record.get("model_id"), metadata["model"])
                values[(workload, arm, round_id)] = float(record[metric])

    stats = {
        f"core_pr_serve.{tag}.model": metadata["model"],
        f"core_pr_serve.{tag}.tp": int(metadata["tp"]),
        f"core_pr_serve.{tag}.base_sha": metadata["base_sha"],
        f"core_pr_serve.{tag}.pr_sha": metadata["pr_sha"],
    }
    print(f"{tag}: {metadata['model']} TP={metadata['tp']} GPU={metadata['gpu']}")
    for workload in WORKLOAD_METRIC:
        for arm in ARMS:
            arm_values = [values[(workload, arm, r)] for r in ROUNDS]
            stats[f"core_pr_serve.{tag}.{workload}.{arm}_median"] = st.median(arm_values)
        ratios = [values[(workload, "pr", r)] / values[(workload, "base", r)] for r in ROUNDS]
        wins = sum(ratio > 1 for ratio in ratios)
        stats[f"core_pr_serve.{tag}.{workload}.pr_vs_base_median"] = st.median(ratios) - 1
        stats[f"core_pr_serve.{tag}.{workload}.pr_wins"] = wins
        stats[f"core_pr_serve.{tag}.{workload}.rounds"] = len(ROUNDS)
        stats[f"core_pr_serve.{tag}.{workload}.pr_vs_base_min"] = min(ratios) - 1
        stats[f"core_pr_serve.{tag}.{workload}.pr_vs_base_max"] = max(ratios) - 1
        for r, ratio in zip(ROUNDS, ratios):
            stats[f"core_pr_serve.{tag}.{workload}.pr_vs_base_r{r}"] = ratio - 1
        for arm in ("base", "pr"):
            ratios_to_off = [values[(workload, arm, r)] / values[(workload, "off", r)] for r in ROUNDS]
            stats[f"core_pr_serve.{tag}.{workload}.{arm}_loss_vs_off"] = 1 - st.median(ratios_to_off)
        metric = WORKLOAD_METRIC[workload]
        print(
            f"  {workload} ({metric}): off {stats[f'core_pr_serve.{tag}.{workload}.off_median']:.1f}, "
            f"base {stats[f'core_pr_serve.{tag}.{workload}.base_median']:.1f}, "
            f"PR {stats[f'core_pr_serve.{tag}.{workload}.pr_median']:.1f}; "
            f"PR/base {stats[f'core_pr_serve.{tag}.{workload}.pr_vs_base_median']:+.1%} "
            f"({wins}/{len(ROUNDS)} wins), range "
            f"[{stats[f'core_pr_serve.{tag}.{workload}.pr_vs_base_min']:+.1%}, "
            f"{stats[f'core_pr_serve.{tag}.{workload}.pr_vs_base_max']:+.1%}]"
        )
    return stats


if "--tags" not in sys.argv:
    raise SystemExit("Usage: score_core_pr_serve.py --tags tag1,tag2 [--merge]")
entries = {}
for tag in sys.argv[sys.argv.index("--tags") + 1].split(","):
    entries.update(score_tag(tag))

if "--merge" in sys.argv:
    sys.path.insert(0, str(ROOT / "src"))
    from analyze import merge

    merge(entries)
    print(f"merged {len(entries)} stats into results/stats.json")
