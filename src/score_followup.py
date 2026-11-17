#!/usr/bin/env python3
"""Score the 2026-11-16 follow-up serving runs (PROTOCOL.md "gpt-oss follow-up to #65931" and
"Diagnostic: Gemma-2-9B prefill regression under PR #67376") from results/validation/oss_followup/<tag>.

Per workload and arm pair a/b: per-round ratios a/b - 1 (output-token throughput for decode workloads, request throughput
for prefill), their median, wins (rounds with a > b) and range. Validity: every bench record completed requests, and the
breakable-graph auto-enable marker appears exactly in the arms listed in --marker-arms.
Usage: score_followup.py --tag <tag> --pairs cfg/base,gate/base --marker-arms gate [--merge]
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "VLLM_BATCH_INVARIANT=1: auto-enabling VLLM_USE_BREAKABLE_CUDAGRAPH=1"
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--pairs", required=True)
ap.add_argument("--marker-arms", default="")
ap.add_argument("--merge", action="store_true")
a = ap.parse_args()

D = ROOT / "results" / "validation" / "oss_followup" / a.tag
meta = json.loads((D / "metadata.json").read_text())
arms = sorted(meta["arms"])
rounds = sorted({int(p.stem.rsplit("_r", 1)[1]) for p in D.glob("bench_*_r*.json")})
workloads = sorted({p.stem.split("_")[1] for p in D.glob("bench_*.json")})
marker_arms = set(filter(None, a.marker_arms.split(",")))
ok = True
for arm in arms:
    for r in rounds:
        has = MARKER in (D / f"server_{arm}_r{r}.log").read_text()
        if has != (arm in marker_arms):
            print(f"INVALID: marker {'present' if has else 'absent'} in {arm} r{r}")
            ok = False
S = {}
P = f"followup.{a.tag}"
print(f"{a.tag}: {meta['model']} on {meta['gpu']}, arms {arms}, rounds {rounds}")
for w in workloads:
    metric = "request_throughput" if w.startswith("prefill") else "output_throughput"
    val = {}
    for arm in arms:
        for r in rounds:
            rec = json.loads((D / f"bench_{w}_{arm}_r{r}.json").read_text())
            assert rec["completed"] > 0, (w, arm, r)
            val[arm, r] = rec[metric]
        S[f"{P}.{w}.{arm}_median"] = st.median(val[arm, r] for r in rounds)
    line = []
    for pair in a.pairs.split(","):
        x, y = pair.split("/")
        rs = [val[x, r] / val[y, r] - 1 for r in rounds]
        k = f"{P}.{w}.{x}_vs_{y}"
        S[k + "_median"], S[k + "_wins"], S[k + "_min"], S[k + "_max"] = st.median(rs), sum(v > 0 for v in rs), min(rs), max(rs)
        for r, v in zip(rounds, rs):
            S[f"{k}_r{r}"] = v
        line.append(f"{x}/{y} {st.median(rs):+.1%} ({sum(v > 0 for v in rs)}/{len(rs)}) [{min(rs):+.1%}, {max(rs):+.1%}]")
    print(f"  {w} ({metric}): " + "; ".join(line))
S[f"{P}.rounds"] = len(rounds)
S[f"{P}.valid"] = ok
print("VALID" if ok else "VOID (marker check failed)")
if a.merge:
    if not ok:
        sys.exit("not merging an invalid run")
    sys.path.insert(0, str(ROOT / "src"))
    from analyze import merge
    merge(S)
    print("merged", len(S), "keys")
