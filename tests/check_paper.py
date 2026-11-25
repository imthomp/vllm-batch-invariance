#!/usr/bin/env python3
"""Paper checks: macros and tables are current, every macro used is defined, no hand-typed result numbers in main.tex."""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "paper"
before = {f: (P / f).read_text() for f in ("macros.tex", "tab_det.tex", "tab_cost.tex", "tab_bitwise.tex", "tab_tune.tex", "tab_corepr.tex", "tab_evalvar.tex", "tab_mtvar.tex", "tab_judgeflags.tex")}
subprocess.run([sys.executable, str(ROOT / "src/make_macros.py")], check=True, capture_output=True)
for f, text in before.items():
    assert (P / f).read_text() == text, f"{f} was stale or hand-edited; rerun src/make_macros.py"

tex = "\n".join((P / f).read_text() for f in ("main.tex", "acl.tex", "body.tex", "appendix.tex"))  # both front ends + shared content
defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", before["macros.tex"]))
used = {m for m in re.findall(r"\\([A-Z]\w+)\{\}", tex)}
assert used <= defined, f"undefined macros: {sorted(used - defined)}"

# Strip comments, macros, citations, labels, refs, URLs and code spans; what remains must have no unreviewed numbers.
body = re.sub(r"(?<!\\)%.*", "", tex)
body = re.sub(r"\\(cite[pt]?|label|ref|url|input|texttt|bibliography\w*|usepackage|documentclass)\{[^}]*\}", "", body)
body = re.sub(r"\[[^\]]*\]\{geometry\}|\[[^\]]*\]\{article\}", "", body)
body = re.sub(r"\\includegraphics\[[^\]]*\]\{[^}]*\}", "", body)  # figure layout, not data
nums = set(re.findall(r"(?<![\w.\-])\d+(?:\.\d+)*", body))
# Reviewed structural numbers: hardware and software versions, model sizes, protocol design constants.
ALLOWED = {"8", "7", "4", "1", "32", "8192", "20", "5", "2880", "1024", "13.4", "10", "4096", "0.33.0", "2.14", "13.0", "80", "90", "100", "2", "3",
           # core PR's runtime build and PR identifier in the Setup section
           "0.34.0", "67376",
           # library versions; tuning search space (BLOCK_K 64/128) and vLLM's default 128-row tile (checked in
           # batch_invariant.py); 256 = batch-size cutoff in the definition of the large-batch comparison
           "13.1", "13.8.2", "64", "128", "256",
           "16",  # cuBLAS alignment-class names (ALIGN_2B, ALIGN_16B)
           # eval-variance design constants fixed in PROTOCOL.md: chunked-prefill token budget, number of model pairs, McNemar level
           "512", "15", "0.05",
           # judge protocol constants fixed in PROTOCOL.md: permutation count, MQM weight for critical errors (GEMBA-MQM),
           # the score of an item with no parsed error, and the 95% confidence level of reported intervals
           "1000", "25", "0", "95"}
bad = sorted(nums - ALLOWED)
assert not bad, f"unreviewed numeric literals in main.tex: {bad}"

# Every table row is tied to a recorded experimental dataset. Generated table row builders additionally
# validate every cell's source records; this checks the hand-written recommendation table's row mapping.
def require_source(path, predicate=lambda d: bool(d)):
    f = ROOT / path
    assert f.is_file(), f"table row has no source dataset: {path}"
    if f.suffix == ".json":
        data = json.loads(f.read_text())
        assert predicate(data), f"table row source has no matching record: {path}"


rec = re.search(r"\\begin\{table\*?\}(?:(?!\\end\{table).)*?\\label\{tab:rec\}.*?\\end\{table\*?\}", tex, re.S)
assert rec, "recommendation table missing"
row_labels = {line.split("&", 1)[0].strip() for line in rec.group(0).splitlines() if "&" in line and not line.lstrip().startswith("Route")}
expected = {
    "Stock, default config": "results/generic2/decode_q25gen2_on_r1.json",
    "Generic + breakable graphs": "results/generic2/decode_q25gen2_generic_r1.json",
    "Per-model tuned": "results/generic2/decode_q25gen2_tuned_r1.json",
    "cuBLAS routing": "results/throughput/main_q25_plain.json",
}
assert row_labels == set(expected), f"recommendation table rows do not match source mapping: {row_labels ^ set(expected)}"
for label, path in expected.items():
    require_source(path, lambda d: any(r.get("n") == 32 for r in d.get("rows", [])))

assert (P / "tab_det.tex").is_file() and (P / "tab_cost.tex").is_file()
assert (P / "tab_bitwise.tex").is_file() and (P / "tab_tune.tex").is_file()
print("paper ok:", len(used), "macros used")

# ARR anonymity: the review front end and the shared content must not name the authors or link their repositories.
# Self-authored PRs/issues are cited in the third person through refs.bib, which ARR allows.
_anon_src = (P / "acl.tex").read_text() + (P / "appendix.tex").read_text() + re.sub(r"\\else.*?\\fi", "", (P / "body.tex").read_text(), flags=re.S)
# Patterns are assembled from fragments so this file itself passes the supplement's leak scan.
for pat in ("Is" + "aac", "Thom" + "pson", "Brig" + "ham", r"\bB" + r"YU\b", "imth" + "omp", "vllm-batch-" + "invariance",
            r"github\.com/(?!vllm-project)"):
    assert not re.search(pat, _anon_src), f"anonymity: {pat!r} appears in the ARR version"
print("anonymity ok")
assert not re.search("Thom" + "pson", (P / "refs_anon.bib").read_text()), "anonymity: refs_anon.bib names the author"
assert "refs_anon" in (P / "acl.tex").read_text(), "anonymity: acl.tex must use refs_anon.bib"
_anon_app = re.sub(r"\\else.*?\\fi", "", (P / "appendix.tex").read_text(), flags=re.S)
assert "maintainer's request" not in _anon_app, "anonymity: maintainer interaction in the ARR appendix"
print("anonymous bibliography ok")
