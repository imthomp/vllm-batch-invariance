#!/usr/bin/env python3
"""Recompute every reported number from results/ and merge it into results/stats.json."""
import collections
import gzip
import itertools
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
STATS = RES / "stats.json"


def steps(f):
    by = {}
    for r in json.load(open(f))["rows"]:
        by.setdefault(r["n"], []).append(r["step_s"])
    return {n: st.median(v) for n, v in by.items()}


def agreement(f):
    g = collections.defaultdict(list)
    with gzip.open(f, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            g[r["i"]].append(r)
    pairs = [(a, b) for v in g.values() for a, b in itertools.combinations(v, 2)]
    return {
        "requests": len(g),
        "pairs": len(pairs),
        "mismatch_pairs": sum(a["t"] != b["t"] for a, b in pairs),
        "mismatch_requests": sum(len({json.dumps(r["t"]) for r in v}) > 1 for v in g.values()),
        "topk_diff_requests": sum(len({json.dumps(r["top"]) for r in v}) > 1 for v in g.values()),
        "min_warm_hit": min(r["hit"] for v in g.values() for r in v if r["cond"].startswith("warm")),
    }


def compute():
    s = {}
    T = RES / "throughput"
    for tag in ("qwen25", "qwen3", "llama31"):
        f = {"qwen25": "q25", "qwen3": "qwen", "llama31": "llama"}[tag]
        off, on, plain = (steps(T / f"main_{f}_{m}.json") for m in ("off", "on", "plain"))
        for n in sorted(off):
            s[f"main.{tag}.b{n}.stock_loss"] = 1 - off[n] / on[n]
            s[f"main.{tag}.b{n}.plain_loss"] = 1 - off[n] / plain[n]
    off, on, hop = (steps(T / f"v032_qwen_{m}.json") for m in ("off", "on", "hop"))
    for n in sorted(off):
        s[f"v032.qwen3.b{n}.stock_loss"] = 1 - off[n] / on[n]
        s[f"v032.qwen3.b{n}.plain_loss"] = 1 - off[n] / hop[n]
    for f in sorted((RES / "crossover").glob("*.jsonl.gz")):
        for k, v in agreement(f).items():
            s[f"cross.{f.name.removesuffix('.jsonl.gz')}.{k}"] = v
    for f in sorted((RES / "bitwise").glob("*.json")):
        rows = json.load(open(f))
        s[f"bitwise.{f.stem}.rows"] = sum(r["rows"] for r in rows)
        s[f"bitwise.{f.stem}.bad"] = sum(r["bad"] for r in rows)
        bad = [r["m"] for r in rows if r["bad"]]
        s[f"bitwise.{f.stem}.min_bad_m"] = min(bad) if bad else None
        s[f"bitwise.{f.stem}.shapes"] = len({(r["k"], r["n"]) for r in rows})
        s[f"bitwise.{f.stem}.bad_shapes"] = len({(r["k"], r["n"]) for r in rows if r["bad"]})
    return s


def merge(entries):
    cur = json.loads(STATS.read_text()) if STATS.exists() else {}
    cur.update(entries)
    STATS.write_text(json.dumps(cur, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    s = compute()
    merge(s)
    print(len(s), "stats written to", STATS)
