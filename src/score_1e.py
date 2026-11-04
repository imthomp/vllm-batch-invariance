#!/usr/bin/env python3
"""Score Phase 1e (cuBLAS's own batch-invariance mode on B200) against PROTOCOL.md "Phase 1e".

Usage: python src/score_1e.py [--from <job output dir>] [--merge]
--from copies the arm JSONs (gzipped) and the job log into results/phase1e first; scoring reads only results/phase1e.
Arms: A cuBLAS 13.1 bi (diagnosis), B 13.8 flags-only control, C 13.8 + CUBLAS_BATCH_INVARIANCE_FLAGS=8 (the test),
D 13.8 + flag with PyTorch defaults, E as C in fp32. An arm is void if its JSON is missing or another cuBLASLt was loaded.
Rule: the mode works if C has 0 mismatches and 0 errors over every shape, dtype and batch size, and B has
mismatches on at least one shape (otherwise the variable is not shown to do anything).
"""
import glob
import gzip
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = Path(os.environ.get("VBI_DST", ROOT / "results" / "phase1e"))
ARMS = {"A": ("A_lib131_bi", 130101), "B": ("B_lib138_bi_noenv", 130802), "C": ("C_lib138_flags8_bi_noenv", 130802),
        "D": ("D_lib138_flags8_default", 130802), "E": ("E_lib138_flags8_fp32", 130802)}

if "--from" in sys.argv:
    src = sys.argv[sys.argv.index("--from") + 1]
    DST.mkdir(parents=True, exist_ok=True)
    for stem, _ in ARMS.values():
        if os.path.exists(f"{src}/{stem}.json"):
            with open(f"{src}/{stem}.json", "rb") as fi, gzip.open(DST / f"{stem}.json.gz", "wb") as fo:
                shutil.copyfileobj(fi, fo)
    for f in glob.glob(f"{src}/bw1e_b200_*.out"):
        shutil.copy(f, DST)

S = {}
arms = {}
for arm, (stem, want) in ARMS.items():
    p = DST / f"{stem}.json.gz"
    if not p.exists():
        arms[arm] = None
        print(f"  {arm}: missing (void)")
        continue
    d = json.load(gzip.open(p))
    versions = {v for _, v in d["cublaslt"]}
    rows = d["rows"]
    bad_shapes = sorted({f"{r['n']}x{r['k']}|{r['dtype']}" for r in rows if r["bad"]})
    err_shapes = sorted({f"{e['n']}x{e['k']}|{e['dtype']}" for e in d.get("errors", [])})
    a = {"void": versions != {want} or "B200" not in d["gpu"], "gpu": d["gpu"], "cublaslt": sorted(versions), "flags": d.get("cublas_bi_flags"),
         "comparisons": sum(r["rows"] for r in rows), "bad": sum(r["bad"] for r in rows),
         "shapes": len({(r["n"], r["k"], r["dtype"]) for r in rows} | {(e["n"], e["k"], e["dtype"]) for e in d.get("errors", [])}),
         "bad_shapes": bad_shapes, "error_shapes": err_shapes}
    arms[arm] = a
    for k, v in a.items():
        S[f"p1e.{arm}.{k}"] = v
    print(f"  {arm}: {'VOID ' if a['void'] else ''}{a['gpu']} cuBLASLt {a['cublaslt']} flags={a['flags']}  "
          f"bad {a['bad']}/{a['comparisons']}  bad shapes {len(bad_shapes)}  errors {len(err_shapes)}")
    if bad_shapes:
        print(f"     bad: {bad_shapes[:20]}")
    if err_shapes:
        print(f"     errors: {err_shapes[:20]}")

B, C = arms.get("B"), arms.get("C")
if not B or not C or B["void"] or C["void"]:
    verdict = "void"
elif C["bad"] == 0 and not C["error_shapes"] and B["bad"] > 0:
    verdict = "works"
elif C["bad"] == 0 and not C["error_shapes"]:
    verdict = "not_shown"  # C clean but the control was clean too: the variable is not shown to do anything
else:
    verdict = "fails"
S["p1e.verdict"] = verdict
print(f"\nPhase 1e verdict: {verdict}")

if "--merge" in sys.argv:
    sys.path.insert(0, str(ROOT / "src"))
    from analyze import merge
    merge(S)
    print("merged", len(S), "keys")
