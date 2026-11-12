#!/usr/bin/env python3
"""Score the better-default experiment against PROTOCOL.md "Better default".

Usage: python src/score_generic.py [--from <job output dir>] [--merge]
--from copies generic.json, the decode JSONs/logs, the crossover JSONL (gzipped) and its log into results/generic.
(1) held-out bitwise check: 0 rows differ; (2) Qwen2.5-7B crossover with the generic default: 0 of 30 requests mismatch,
generic marker logged; (3) Qwen2.5-7B decode loss at batch 32 (1 - t_off / t_mode, medians over three jobs): the generic
default recovers at least 75% of the stock-to-tuned gap, (stock - generic) / (stock - tuned) >= 0.75.
"""
import collections
import glob
import gzip
import json
import os
import shutil
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = Path(os.environ.get("VBI_DST", ROOT / "results" / "generic"))
if "--from" in sys.argv:
    src = sys.argv[sys.argv.index("--from") + 1]
    DST.mkdir(parents=True, exist_ok=True)
    for f in glob.glob(f"{src}/generic.json") + glob.glob(f"{src}/decode_*") + glob.glob(f"{src}/cross_*.log") + glob.glob(f"{src}/*.out"):
        shutil.copy(f, DST)
    for f in glob.glob(f"{src}/cross_*.jsonl"):
        with open(f, "rb") as fi, gzip.open(DST / (os.path.basename(f) + ".gz"), "wb") as fo:
            shutil.copyfileobj(fi, fo)

S, ok = {}, {}
g = json.loads((DST / "generic.json").read_text())
S["generic.block_k"] = g["generic"]["block_k"]
S["generic.train_shapes"], S["generic.test_shapes"] = len(g["train"]), len(g["test"])
S["generic.check_bad"] = sum(v["bad"] for v in g["check"].values())
S["generic.check_rows"] = sum(v["rows"] for v in g["check"].values())
sp = [v[m]["default"] / v[m]["generic"] for v in g["test_speed"].values() for m in v]
small = [v[m]["default"] / v[m]["generic"] for v in g["test_speed"].values() for m in v if int(m) <= 32]
S["generic.test_speed_min"], S["generic.test_speed_max"] = min(sp), max(sp)
S["generic.test_speed_small_min"], S["generic.test_speed_small_max"] = min(small), max(small)
ok[1] = S["generic.check_bad"] == 0 and S["generic.test_shapes"] == len(g["check"])
print(f"(1) held-out bitwise: {S['generic.check_bad']} of {S['generic.check_rows']} rows differ over {len(g['check'])} shapes "
      f"{'PASS' if ok[1] else 'FAIL'}; speedup over default {min(sp):.2f}-{max(sp):.2f}x (batch <= 32: {min(small):.2f}-{max(small):.2f}x)")

grp = collections.defaultdict(list)
for line in gzip.open(DST / "cross_q25gen_generic.jsonl.gz", "rt"):
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        continue
    grp[r["i"]].append(r)
full = {i: v for i, v in grp.items() if len(v) == 8}
log = (DST / "cross_q25gen_generic.log").read_text()
S["generic.cross_requests"] = len(full)
S["generic.cross_mismatch"] = sum(len({json.dumps(r["t"]) for r in v}) > 1 for v in full.values())
S["generic.cross_topk_diff"] = sum(len({json.dumps(r["top"]) for r in v}) > 1 for v in full.values())
S["generic.cross_markers"] = log.count("patched generic default")
ok[2] = S["generic.cross_requests"] == 30 and S["generic.cross_mismatch"] == 0 and S["generic.cross_markers"] >= 1 and "DONE requests" in log
print(f"(2) crossover: {S['generic.cross_mismatch']} of {S['generic.cross_requests']} mismatch, top-k differ {S['generic.cross_topk_diff']}, "
      f"markers {S['generic.cross_markers']} {'PASS' if ok[2] else 'FAIL'}")


def steps(tag, mode):
    reps = []
    for f in sorted(DST.glob(f"decode_{tag}_{mode}_r*.json")):
        rows = json.load(open(f))["rows"]
        reps.append({n: st.median(r["step_s"] for r in rows if r["n"] == n) for n in sorted({r["n"] for r in rows})})
    return {n: st.median(r[n] for r in reps) for n in reps[0]}, len(reps)


for tag, modes in (("q25gen", ("on", "tuned", "generic")), ("g2gen", ("on", "generic"))):
    off, n_off = steps(tag, "off")
    for mode in modes:
        t, n = steps(tag, mode)
        for b in off:
            S[f"generic.{tag}.b{b}.{mode}_loss"] = 1 - off[b] / t[b]
        S[f"generic.{tag}.{mode}_n"] = n
    S[f"generic.{tag}.off_n"] = n_off
    print(f"    {tag}: " + "  ".join(f"b{b}: " + " ".join(f"{m} {S[f'generic.{tag}.b{b}.{m}_loss']:.1%}" for m in modes) for b in off))
st_, tu, ge = (S[f"generic.q25gen.b32.{m}_loss"] for m in ("on", "tuned", "generic"))
S["generic.q25.b32.recovered"] = (st_ - ge) / (st_ - tu)
ok[3] = S["generic.q25.b32.recovered"] >= 0.75
print(f"(3) Qwen2.5-7B batch 32: stock {st_:.1%}, tuned {tu:.1%}, generic {ge:.1%} -> recovers {S['generic.q25.b32.recovered']:.0%} of the gap "
      f"{'PASS' if ok[3] else 'FAIL'}")
print(f"    Gemma-2-9B batch 32: stock {S['generic.g2gen.b32.on_loss']:.1%}, generic {S['generic.g2gen.b32.generic_loss']:.1%}")
S["generic.pass"] = all(ok.values())
print("\nBetter default verdict:", "PASS" if S["generic.pass"] else "FAIL", {k: v for k, v in ok.items()})
if "--merge" in sys.argv:
    sys.path.insert(0, str(ROOT / "src"))
    from analyze import merge
    merge(S)
    print("merged", len(S), "keys")
