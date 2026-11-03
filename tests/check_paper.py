#!/usr/bin/env python3
"""Paper checks: macros and tables are current, every macro used is defined, no hand-typed result numbers in main.tex."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "paper"
before = {f: (P / f).read_text() for f in ("macros.tex", "tab_det.tex", "tab_cost.tex", "tab_bitwise.tex")}
subprocess.run([sys.executable, str(ROOT / "src/make_macros.py")], check=True, capture_output=True)
for f, text in before.items():
    assert (P / f).read_text() == text, f"{f} was stale or hand-edited; rerun src/make_macros.py"

tex = (P / "main.tex").read_text()
defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", before["macros.tex"]))
used = {m for m in re.findall(r"\\([A-Z]\w+)\{\}", tex)}
assert used <= defined, f"undefined macros: {sorted(used - defined)}"

# Strip comments, macros, citations, labels, refs, URLs and code spans; what remains must have no unreviewed numbers.
body = re.sub(r"(?<!\\)%.*", "", tex)
body = re.sub(r"\\(cite[pt]?|label|ref|url|input|texttt|bibliography\w*|usepackage|documentclass)\{[^}]*\}", "", body)
body = re.sub(r"\[[^\]]*\]\{geometry\}|\[[^\]]*\]\{article\}", "", body)
nums = set(re.findall(r"(?<![\w.\-])\d+(?:\.\d+)*", body))
# Reviewed structural numbers: hardware and software versions, model sizes, protocol design constants.
ALLOWED = {"8", "7", "4", "1", "32", "8192", "20", "5", "0.33.0", "2.14", "13.0", "80", "90", "100", "2", "3"}
bad = sorted(nums - ALLOWED)
assert not bad, f"unreviewed numeric literals in main.tex: {bad}"
print("paper ok:", len(used), "macros used")
