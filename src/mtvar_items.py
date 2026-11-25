#!/usr/bin/env python3
"""Build the LLM-judge items (PROTOCOL.md MT study): each system's BI-on C2 r1 translation of the first 250 good-source
segments (by segment_id) of de_DE, ja_JP, is_IS and sw_KE. Usage: mtvar_items.py --root <translate dir> --out items.jsonl [--n 250]"""
import argparse
import json
from pathlib import Path

SYSTEMS = ["gemma2_9b", "gemma3_12b", "aya_8b", "qwen25_7b", "llama31_8b"]
LPS = ["de_DE", "ja_JP", "is_IS", "sw_KE"]
ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=250)
ap.add_argument("--systems", default=",".join(SYSTEMS))
a = ap.parse_args()
items = []
for s in a.systems.split(","):
    recs = [json.loads(x) for x in open(Path(a.root) / s / "bi1" / "C2_r1.jsonl")]
    for lp in LPS:
        seg = sorted((r for r in recs if r["lp"] == lp), key=lambda r: int(r["id"].split(":")[1]))[: a.n]
        assert len(seg) == a.n, (s, lp, len(seg))
        items += [{"id": f"{s}|{r['id']}", "sys": s, "lp": lp, "src": r["src"], "hyp": r["text"].strip()} for r in seg]
Path(a.out).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items))
print(len(items), "judge items ->", a.out)
