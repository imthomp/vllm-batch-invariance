#!/usr/bin/env python3
"""Archive MT eval-variance runs into results/validation/mtvar/<task>/<model>/bi<b>/<run>.jsonl.gz (+ meta, log), with
token_ids replaced by token_sha (sha256 of the id list) and n_tokens, as in src/evalvar_archive.py. Refuses pilot runs.
Usage: mtvar_archive.py [--src /nobackup/.../data/mtvar] [--dest <subdir of results/validation/mtvar> --items <file>]
Extension runs (PROTOCOL.md judge-robustness extension) go to their own --dest, e.g. ext_j1, so they never overwrite the
original judge archive; --items copies that configuration's judge item file next to them."""
import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/nobackup/autodelete/usr/it238/research_lab/data/mtvar")
ap.add_argument("--dest", help="subdirectory of results/validation/mtvar for an extension configuration")
ap.add_argument("--items", help="judge item file to archive with an extension configuration")
a = ap.parse_args()
dest, n = ROOT / "results" / "validation" / "mtvar", 0
if a.dest:
    dest = dest / a.dest
    assert dest.name.startswith(("ext_", "j4_", "j5_")), "extension archives use their own prefixed directory"
for f in sorted(Path(a.src).glob("*/*/bi*/*.jsonl")):
    meta = f.with_suffix(".meta.json")
    m = json.loads(meta.read_text())
    assert not m["limit"], f
    out = dest / f.relative_to(a.src).parent
    out.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(x) for x in f.open()]
    assert len(recs) == m["n_items"], f
    with gzip.open(out / (f.name + ".gz"), "wt") as g:
        for r in recs:
            ids = r.pop("token_ids")
            for k in ("src", "ref", "hyp"):  # inputs are archived once (wmt24pp/, judge_items.jsonl), not per run
                r.pop(k, None)
            r["token_sha"], r["n_tokens"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest(), len(ids)
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    shutil.copy(meta, out / meta.name)
    shutil.copy(f.with_suffix(".log"), out / (f.stem + ".log"))
    n += 1
if a.dest:
    if a.items:
        shutil.copy(a.items, dest / Path(a.items).name)
else:
    shutil.copy(Path(a.src) / "judge_items.jsonl", dest / "judge_items.jsonl")
    shutil.copytree(Path(a.src).parent / "wmt24pp", dest / "wmt24pp", dirs_exist_ok=True)
print("archived", n, "runs to", dest)
