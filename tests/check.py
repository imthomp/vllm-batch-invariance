#!/usr/bin/env python3
"""Checks: stats recompute from raw files, README matches its template, no hand-typed result numbers in the template."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import analyze
import render

stored = json.loads((ROOT / "results/stats.json").read_text())
fresh = analyze.compute()
bad = [k for k, v in fresh.items() if stored.get(k) != v]
assert not bad, f"stats.json out of date: {bad[:5]}"
assert (ROOT / "README.md").read_text() == render.render(), "README.md is stale; run src/render.py"

# Every number in the template prose must be a placeholder, or on this reviewed list of structural numbers
# (section numbers, versions, model sizes, PR numbers, dataset names, protocol constants).
ALLOWED = {
    "2.5", "7", "8", "3", "3.1", "32", "1", "512", "10", "0.32", "0.32.0", "0.33.0", "1.dev454", "2.13", "2.14",
    "13.0", "2", "4", "5", "9", "8192", "500", "80", "2880", "35404", "65569", "1024", "16", "13.4", "20", "90", "100", "256", "2025", "62219", "61118",
}
text = re.sub(r"\{\{[^}]+\}\}", "", (ROOT / "README.tmpl.md").read_text())
text = re.sub(r"```.*?```", "", text, flags=re.S)
text = re.sub(r"`[^`]*`", "", text)
nums = set(re.findall(r"(?<![\w.])\d+(?:\.\d+)*(?:\.dev\d+)?", text))
nums = {n for n in nums if n not in ALLOWED}
assert not nums, f"unreviewed numeric literals in README.tmpl.md: {sorted(nums)}"

# Every throughput and crossover file traces to a stat, and every stat to a file.
files = {p.name.split(".")[0] for p in (ROOT / "results/crossover").glob("*.jsonl.gz")}
assert files == {k.split(".")[1] for k in fresh if k.startswith("cross.")}
print("ok:", len(fresh), "stats")
