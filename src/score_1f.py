#!/usr/bin/env python3
"""Score Phase 1f (PROTOCOL.md): which setting breaks cuBLAS's batch-invariance mode on odd widths. Descriptive.
Usage: python src/score_1f.py [--from <job dir>] [--merge]"""
import glob, gzip, json, os, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DST = Path(os.environ.get("VBI_DST", ROOT / "results" / "phase1f"))
if "--from" in sys.argv:
    src = sys.argv[sys.argv.index("--from") + 1]; DST.mkdir(parents=True, exist_ok=True)
    for f in glob.glob(f"{src}/F_*.json"):
        with open(f, "rb") as fi, gzip.open(DST / (os.path.basename(f) + ".gz"), "wb") as fo: shutil.copyfileobj(fi, fo)
    for f in glob.glob(f"{src}/bw1f_*.out"): shutil.copy(f, DST)
S = {}
for p in sorted(DST.glob("F_*.json.gz")):
    arm = p.name[2:-len(".json.gz")]; d = json.load(gzip.open(p))
    assert {v for _, v in d["cublaslt"]} == {130802} and "B200" in d["gpu"], f"void arm {arm}"
    r = d["rows"]; e = d.get("errors", [])
    S.update({f"p1f.{arm}.comparisons": sum(x["rows"] for x in r), f"p1f.{arm}.bad": sum(x["bad"] for x in r),
              f"p1f.{arm}.bad_shapes": sorted({f"{x['n']}x{x['k']}|{x['dtype']}" for x in r if x["bad"]}),
              f"p1f.{arm}.errors": len(e), f"p1f.{arm}.flags": d.get("cublas_bi_flags"), f"p1f.{arm}.torch_flags": d.get("torch_flags")})
    print(f"{arm:8s} flags={d.get('cublas_bi_flags')} torch={d.get('torch_flags')}  bad {S[f'p1f.{arm}.bad']}/{S[f'p1f.{arm}.comparisons']}  errors {len(e)}  bad shapes {S[f'p1f.{arm}.bad_shapes'][:6]}")
if "--merge" in sys.argv:
    sys.path.insert(0, str(ROOT / "src")); from analyze import merge; merge(S); print("merged", len(S))
