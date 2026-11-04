#!/usr/bin/env python3
"""Score the tuning pilot against PROTOCOL.md "Tuning pilot: does tuning close the Hopper gap?".

Usage: python src/score_tune.py [--from <job output dir>] [--tag q25] [--merge]
--from copies the tuner output, the crossover JSONL (gzipped) with its log, and the decode JSONs/logs into
results/tune first; scoring reads only results/tune.
Pass requires all of: (1) 0 rows differ in tune_bi.py --check; (2) 0 of 30 crossover requests mismatch, run complete,
tuned-table line logged at least once (TP=1); (3) decode loss at batch 32, 1 - t_off / t_tuned, at most 5%.
"""
import collections
import glob
import gzip
import itertools
import json
import os
import shutil
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = Path(os.environ.get("VBI_DST", ROOT / "results" / "tune"))
tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "q25"

if "--from" in sys.argv:
    src = sys.argv[sys.argv.index("--from") + 1]
    DST.mkdir(parents=True, exist_ok=True)
    for f in glob.glob(f"{src}/tuned_{tag}.json") + glob.glob(f"{src}/decode_{tag}_*") + glob.glob(f"{src}/*.out") \
            + glob.glob(f"{src}/cross_{tag}_tuned.log"):
        shutil.copy(f, DST)
    for f in glob.glob(f"{src}/cross_{tag}_tuned.jsonl"):
        with open(f, "rb") as fi, gzip.open(DST / (os.path.basename(f) + ".gz"), "wb") as fo:
            shutil.copyfileobj(fi, fo)

S = {}
ok = {}

# (1) bitwise check of the tuned table, plus per-shape kernel times from the tuner
t = json.load(open(DST / f"tuned_{tag}.json"))
chk = t.get("check", {})
S[f"tune.{tag}.check_bad"] = sum(v["bad"] for v in chk.values())
S[f"tune.{tag}.check_rows"] = sum(v["rows"] for v in chk.values())
S[f"tune.{tag}.shapes"] = len(t["entries"])
ok[1] = bool(chk) and len(chk) == len(t["entries"]) and S[f"tune.{tag}.check_bad"] == 0
print(f"(1) bitwise: {S[f'tune.{tag}.check_bad']} of {S[f'tune.{tag}.check_rows']} rows differ over {len(chk)} shapes  "
      f"{'PASS' if ok[1] else 'FAIL'}")
for e in t["entries"]:
    ms = e["ms"]
    sp = {m: ms[m]["default"] / ms[m]["tuned"] for m in ms}
    vs = {m: ms[m]["tuned"] / ms[m]["cublas"] for m in ms}
    S[f"tune.{tag}.{e['n']}x{e['k']}.block_k"] = e["block_k"]
    S[f"tune.{tag}.{e['n']}x{e['k']}.speedup_vs_default_m1"] = sp.get("1")
    S[f"tune.{tag}.{e['n']}x{e['k']}.ratio_to_cublas_m1"] = vs.get("1")
    print(f"    {e['n']}x{e['k']} BLOCK_K={e['block_k']}  speedup over default: "
          + " ".join(f"M{m}:{v:.2f}x" for m, v in sp.items()) + "  tuned/cuBLAS: "
          + " ".join(f"M{m}:{v:.2f}" for m, v in vs.items()))

# (2) crossover with the tuned table loaded in vLLM
g = collections.defaultdict(list)
cf = DST / f"cross_{tag}_tuned.jsonl.gz"
for line in gzip.open(cf, "rt"):
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    g[r["i"]].append(r)
full = {i: v for i, v in g.items() if len(v) == 8}
log = (DST / f"cross_{tag}_tuned.log").read_text() if (DST / f"cross_{tag}_tuned.log").exists() else ""
S[f"tune.{tag}.cross_requests"] = len(full)
S[f"tune.{tag}.cross_mismatch_requests"] = sum(len({json.dumps(r["t"]) for r in v}) > 1 for v in full.values())
S[f"tune.{tag}.cross_topk_diff_requests"] = sum(len({json.dumps(r["top"]) for r in v}) > 1 for v in full.values())
S[f"tune.{tag}.cross_tuned_markers"] = log.count("patched tuned table")
S[f"tune.{tag}.cross_done"] = "DONE requests" in log
ok[2] = (S[f"tune.{tag}.cross_requests"] == 30 and S[f"tune.{tag}.cross_mismatch_requests"] == 0
         and S[f"tune.{tag}.cross_tuned_markers"] >= 1 and S[f"tune.{tag}.cross_done"])
print(f"(2) crossover: {S[f'tune.{tag}.cross_mismatch_requests']} of {S[f'tune.{tag}.cross_requests']} requests mismatch, "
      f"top-k differ {S[f'tune.{tag}.cross_topk_diff_requests']}, tuned markers {S[f'tune.{tag}.cross_tuned_markers']}, "
      f"done {S[f'tune.{tag}.cross_done']}  {'PASS' if ok[2] else 'FAIL'}")


# (3) decode cost, off measured in the same job
def step(mode):
    f = DST / f"decode_{tag}_{mode}_r1.json"
    rows = json.load(open(f))["rows"]
    return {n: st.median(r["step_s"] for r in rows if r["n"] == n) for n in sorted({r["n"] for r in rows})}


off, on, tuned = step("off"), step("on"), step("tuned")
for n in off:
    S[f"tune.{tag}.b{n}.stock_loss"] = 1 - off[n] / on[n]
    S[f"tune.{tag}.b{n}.tuned_loss"] = 1 - off[n] / tuned[n]
ok[3] = S[f"tune.{tag}.b32.tuned_loss"] <= 0.05
print("(3) decode loss vs off:  " + "  ".join(f"b{n}: stock {S[f'tune.{tag}.b{n}.stock_loss']:.1%} tuned {S[f'tune.{tag}.b{n}.tuned_loss']:.1%}"
                                         for n in off) + f"   b32 {'PASS' if ok[3] else 'FAIL'}")

S[f"tune.{tag}.pass"] = all(ok.values())
verdict = "closes the gap" if all(ok.values()) else ("invariance broken" if not (ok[1] and ok[2]) else "gap remains")
S[f"tune.{tag}.verdict"] = verdict
print(f"\nTuning pilot verdict: {verdict}")

if "--merge" in sys.argv:
    sys.path.insert(0, str(ROOT / "src"))
    from analyze import merge
    merge(S)
    print("merged", len(S), "keys")
