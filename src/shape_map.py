#!/usr/bin/env python3
"""Phase 1c: per-shape batch invariance from bitwise2.py output. Usage: shape_map.py <dir> [mode]"""
import glob
import json
import os
import sys
from collections import defaultdict

d, mode = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "bi")
cells = defaultdict(lambda: {"bad": 0, "rows": 0, "bad_m": []})
for f in sorted(g for g in glob.glob(f"{d}/*_{mode}.json") if not os.path.basename(g).startswith("shape_map")):
    for r in json.load(open(f))["rows"]:
        c = cells[(r["n"], r["k"], r["dtype"])]
        c["bad"] += r["bad"]
        c["rows"] += r["rows"]
        if r["bad"]:
            c["bad_m"].append(r["m"])
print(f"mode={mode}  shape (N, K)  dtype  bad/rows  batch sizes failing  first failing")
summary = {}
for (n, k, dt), c in sorted(cells.items()):
    nm = len(set(c["bad_m"]))
    first = min(c["bad_m"]) if c["bad_m"] else None
    summary[f"{n}x{k}|{dt}"] = {"bad": c["bad"], "rows": c["rows"], "bad_batch_sizes": nm, "first_bad_m": first}
    flag = "FAIL" if c["bad"] else "ok"
    print(f"  {n:>6}x{k:<6} {dt:9s} {c['bad']:>7}/{c['rows']:<8} {nm:>5}  {first}  {flag}")
json.dump(summary, open(os.path.join(d, f"shape_map_{mode}.json"), "w"), indent=1)
failing_n = sorted({int(key.split('x')[0]) for key, v in summary.items() if v["bad"]})
passing_n = sorted({int(key.split('x')[0]) for key, v in summary.items() if not v["bad"]} - set(failing_n))
print("N values with any failure:", failing_n)
print("N values that always pass:", passing_n)
