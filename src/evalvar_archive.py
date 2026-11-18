#!/usr/bin/env python3
"""Archive eval-variance runs into the repo: results/validation/evalvar/<model>/bi<b>/<run>.jsonl.gz (+ .meta.json).
Each record keeps id, src, gold, text and finish, and replaces token_ids with token_sha (sha256 of the id list) and
n_tokens, which is all the batch-invariance identity check needs. Refuses smoke runs and incomplete runs.
Usage: evalvar_archive.py [--src /nobackup/.../data/evalvar]
"""
import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/nobackup/autodelete/usr/it238/research_lab/data/evalvar")
a = ap.parse_args()
dest = ROOT / "results" / "validation" / "evalvar"
n = 0
for f in sorted(Path(a.src).glob("*/bi*/*.jsonl")):
    meta = f.with_suffix(".meta.json")
    if not meta.exists():
        print("skip (no meta, incomplete):", f)
        continue
    m = json.loads(meta.read_text())
    assert not m["limit"], f"smoke run in full-run directory: {f}"
    out = dest / f.parent.parent.name / f.parent.name
    out.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(line) for line in f.open()]
    assert len(recs) == m["n_bench"] == 1819, f
    with gzip.open(out / (f.name + ".gz"), "wt") as g:
        for r in recs:
            ids = r.pop("token_ids")
            r["token_sha"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            r["n_tokens"] = len(ids)
            g.write(json.dumps(r) + "\n")
    shutil.copy(meta, out / meta.name)
    shutil.copy(f.with_suffix(".log"), out / (f.stem + ".log"))
    n += 1
print("archived", n, "runs to", dest)
