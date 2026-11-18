#!/usr/bin/env python3
"""Build the anonymized ARR supplementary archive (code, protocol, aggregated results) from the committed tree.

Raw server/benchmark logs are left out (size; they hold user paths); they are public with the named repository after
review. Identifying strings are replaced, and the build fails if any remain.
Usage: build_supplement.py [--out /tmp/supplement.zip]
"""
import argparse
import io
import re
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEEP = re.compile(r"^(src/|slurm/[^/]+\.(sbatch|sh|env|json|txt)$|tests/|PROTOCOL\.md$|README\.md$|LICENSE$|"
                  r"results/stats\.json$|paper/(acl\.tex|body\.tex|macros\.tex|tab_\w+\.tex|refs\.bib|fig_\w+\.pdf|acl\.sty|acl_natbib\.bst)$)")
SUBS = [(re.compile(p, re.I), r) for p, r in [
    (r"Isaac M\.? Thompson|Isaac Thompson", "Anonymous"), (r"Thompson", "Anonymous"), (r"Isaac", "Anonymous"),
    (r"isaacmthomp@\S+|it238@byu\.edu", "anonymous@example.org"), (r"imthomp", "anonymous"),
    (r"/home/it238|/nobackup/\w+/usr/it238", "/path/to/user"), (r"it238", "user"),
    (r"Brigham Young University", "Anonymous Institution"), (r"\bBYU\b", "ANON"),
    (r"vllm-batch-invariance", "this-repository"), (r"Eric Ringger|Ringger", "Anonymous")]]
LEAK = re.compile(r"isaac|thompson|imthomp|it238|brigham|\bbyu\b|ringger", re.I)

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/tmp/supplement.zip")
a = ap.parse_args()
files = [f for f in subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True,
                                   check=True).stdout.split("\n") if f and KEEP.match(f) and f != "src/build_supplement.py"]
leaks, size = [], 0
with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        data = (ROOT / f).read_bytes()
        if f.endswith(".pdf"):
            z.writestr("supplement/" + f, data)
            continue
        text = data.decode("utf-8")
        for p, r in SUBS:
            text = p.sub(r, text)
        leaks += [f"{f}: {m.group(0)}" for m in LEAK.finditer(text)]
        z.writestr("supplement/" + f, text)
        size += len(text)
assert not leaks, "identifying strings remain:\n" + "\n".join(leaks[:20])
print(f"{len(files)} files, {size / 1e6:.1f} MB uncompressed, {Path(a.out).stat().st_size / 1e6:.1f} MB zip -> {a.out}")
