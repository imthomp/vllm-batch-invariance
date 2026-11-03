#!/usr/bin/env python3
"""Score Phase 1 and Phase 2 runs against PROTOCOL.md.

Usage: python src/phase_score.py <phase1 dir> <phase2 dir> [--merge]
Reads bitwise JSON files (phase 1), crossover JSONL plus logs, and decode/prefill JSON (phase 2).
--merge writes the numbers into results/stats.json under phase1.* and phase2.*.
"""
import collections
import glob
import gzip
import itertools
import json
import os
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import merge

p1, p2 = sys.argv[1], sys.argv[2]
S = {}

print("== Phase 1: bitwise (pass = 0 bad in bi mode; control must be > 0)")
for f in sorted(glob.glob(f"{p1}/*_bi.json")):
    tag = os.path.basename(f)[:-len("_bi.json")]
    row = []
    for mode in ("bi", "bi_noenv", "default"):
        g = f"{p1}/{tag}_{mode}.json"
        if not os.path.exists(g):
            row.append(f"{mode}=missing")
            continue
        d = json.load(open(g))
        bad, n = sum(r["bad"] for r in d["rows"]), sum(r["rows"] for r in d["rows"])
        S[f"phase1.{tag}.{mode}.bad"], S[f"phase1.{tag}.{mode}.rows"] = bad, n
        S[f"phase1.{tag}.cublaslt"] = str(d.get("cublaslt"))
        row.append(f"{mode}={bad}/{n}")
    print(f"  {tag:34s} " + " ".join(row))


def cross(f):
    g = collections.defaultdict(list)
    for line in (gzip.open(f, "rt") if f.endswith(".gz") else open(f)):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        g[r["i"]].append(r)
    full = {i: v for i, v in g.items() if len(v) == 8}
    pairs = [(a, b) for v in full.values() for a, b in itertools.combinations(v, 2)]
    stem = f[:-len(".jsonl.gz")] if f.endswith(".gz") else f[:-len(".jsonl")]
    log = next((p for p in (stem + ".log", stem + ".markers.txt") if os.path.exists(p)), None)
    text = open(log).read() if log else ""
    warm = [r["hit"] for v in full.values() for r in v if r["cond"].startswith("warm")]
    return {
        "requests": len(full),
        "done": "DONE requests" in text,
        "markers": text.count("patched linear_batch_invariant"),
        "pairs": len(pairs),
        "mismatch_pairs": sum(a["t"] != b["t"] for a, b in pairs),
        "mismatch_requests": sum(len({json.dumps(r["t"]) for r in v}) > 1 for v in full.values()),
        "topk_diff_requests": sum(len({json.dumps(r["top"]) for r in v}) > 1 for v in full.values()),
        "min_warm_hit": min(warm) if warm else None,
    }


print("\n== Phase 2: determinism (pass = 0 mismatches, markers >= TP, warm hit >= 0.9; control must diverge)")
for f in sorted(glob.glob(f"{p2}/cross_*.jsonl") + glob.glob(f"{p2}/cross_*.jsonl.gz")):
    name = os.path.basename(f)[len("cross_"):].split(".jsonl")[0]
    c = cross(f)
    for k, v in c.items():
        S[f"phase2.cross.{name}.{k}"] = v
    print(f"  {name:22s} " + " ".join(f"{k}={v}" for k, v in c.items()))


def median_loss(kind, tag, n_or_cell):
    out = {}
    for mode in ("off", "on", "plain"):
        vals = []
        for f in sorted(glob.glob(f"{p2}/{kind}_{tag}_{mode}_r*.json")):
            d = json.load(open(f))
            if kind == "decode":
                v = st.median(r["step_s"] for r in d["rows"] if r["n"] == n_or_cell)
            else:
                ln, n = n_or_cell
                v = st.median(r["s_per_tok"] for r in d["rows"] if r["len"] == ln and r["n"] == n)
            vals.append(v)
        out[mode] = vals
    return out


print("\n== Phase 2: cost (loss = 1 - t_off / t_mode; medians over repeats; pass = plain <= 5%, stock - plain >= 10 pts on untuned decode)")
for kind, tags, cells in (("decode", ("q25", "mistral", "gemma2", "q25tp2", "q32tp2"), (32,)),
                          ("prefill", ("q25", "mistral"), [(ln, n) for ln in (1024, 4096) for n in (1, 8, 32)])):
    for tag in tags:
        for cell in cells:
            t = median_loss(kind, tag, cell)
            if not t["off"]:
                continue
            off = st.median(t["off"])
            parts = []
            for mode in ("on", "plain"):
                if t[mode]:
                    losses = [1 - off / v for v in t[mode]]
                    key = f"phase2.{kind}.{tag}.{re.sub(r'[^0-9a-z]+', '_', str(cell)).strip('_')}.{mode}"
                    S[key + "_loss_median"] = 1 - off / st.median(t[mode])
                    S[key + "_loss_max"] = max(losses)
                    S[key + "_n"] = len(t[mode])
                    parts.append(f"{mode} {S[key + '_loss_median']:.1%} (max {max(losses):.1%}, n={len(t[mode])})")
            print(f"  {kind:7s} {tag:8s} {str(cell):12s} off n={len(t['off'])}  " + "  ".join(parts))

if "--merge" in sys.argv:
    merge(S)
    print("\nmerged", len(S), "keys into results/stats.json")
