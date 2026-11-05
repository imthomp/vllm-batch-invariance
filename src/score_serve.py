#!/usr/bin/env python3
"""Score the tuned-config PR validation (PROTOCOL.md "Tuned-config PR validation").

Usage: python src/score_serve.py [--from <job output dir>] [--merge]
--from copies bench_*.json, unit.log and the job log into results/serve first; scoring reads only results/serve.
Pass for the PR claim: tuned output-token throughput beats stock in every round on the decode-heavy workload.
"""
import glob
import json
import os
import re
import shutil
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = Path(os.environ.get("VBI_DST", ROOT / "results" / "serve"))

if "--from" in sys.argv:
    src = sys.argv[sys.argv.index("--from") + 1]
    DST.mkdir(parents=True, exist_ok=True)
    for f in glob.glob(f"{src}/bench_*.json") + glob.glob(f"{src}/unit.log") + glob.glob(f"{src}/tunedpr_*.out"):
        shutil.copy(f, DST)

S = {}
unit = (DST / "unit.log").read_text() if (DST / "unit.log").exists() else ""
m = re.search(r"(\d+) passed", unit)
S["serve.unit_passed"] = int(m.group(1)) if m else 0
S["serve.unit_failed"] = int(re.search(r"(\d+) failed", unit).group(1)) if re.search(r"(\d+) failed", unit) else 0
S["serve.unit_new_passed"] = len(re.findall(r"PASSED .*test_qwen25_7b_hopper_tuned_linear_batch_invariance", unit))
print(f"unit: {S['serve.unit_passed']} passed, {S['serve.unit_failed']} failed, new test cases passed {S['serve.unit_new_passed']}/5")

metric = {"decode": "output_throughput", "prefill": "request_throughput"}
runs = {}
for f in sorted(DST.glob("bench_*_r*.json")):
    w, arm, r = re.match(r"bench_(\w+?)_(off|stock|tuned)_r(\d+)\.json", f.name).groups()
    d = json.load(open(f))
    assert d.get("completed", 0) == d.get("num_prompts", d.get("completed")), f"incomplete run {f.name}"
    runs[(w, arm, int(r))] = d[metric[w]]
rounds = sorted({r for _, _, r in runs})
for w in metric:
    mean = {}
    for arm in ("off", "stock", "tuned"):
        vals = [runs[(w, arm, r)] for r in rounds if (w, arm, r) in runs]
        if vals:
            mean[arm] = st.mean(vals)
            S[f"serve.{w}.{arm}_mean"] = mean[arm]
            S[f"serve.{w}.{arm}_n"] = len(vals)
    if {"stock", "tuned"} <= set(mean):
        wins = sum(runs[(w, "tuned", r)] > runs[(w, "stock", r)] for r in rounds if (w, "tuned", r) in runs and (w, "stock", r) in runs)
        S[f"serve.{w}.tuned_wins"] = wins
        S[f"serve.{w}.rounds"] = len(rounds)
        S[f"serve.{w}.gain_over_stock"] = mean["tuned"] / mean["stock"] - 1
    if "off" in mean:
        for arm in ("stock", "tuned"):
            if arm in mean:
                S[f"serve.{w}.{arm}_loss_vs_off"] = 1 - mean[arm] / mean["off"]
    print(f"{w} ({metric[w]}): " + "  ".join(f"{a} {v:.1f}" for a, v in mean.items())
          + (f"  gain {S[f'serve.{w}.gain_over_stock']:+.1%}, tuned wins {S[f'serve.{w}.tuned_wins']}/{len(rounds)}" if f"serve.{w}.gain_over_stock" in S else "")
          + "".join(f"  {a} loss vs off {S[f'serve.{w}.{a}_loss_vs_off']:.1%}" for a in ("stock", "tuned") if f"serve.{w}.{a}_loss_vs_off" in S))
S["serve.pass"] = S.get("serve.decode.tuned_wins", -1) == len(rounds) and len(rounds) > 0 and S["serve.unit_failed"] == 0
print("PR claim:", "PASS" if S["serve.pass"] else "FAIL")

if "--merge" in sys.argv:
    sys.path.insert(0, str(ROOT / "src"))
    from analyze import merge
    merge(S)
    print("merged", len(S), "keys")
