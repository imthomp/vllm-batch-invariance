#!/usr/bin/env python3
"""MT pilot analysis: output changes across serving runs, metric ranges, and ranking/significance flips."""
import glob
import itertools
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sacrebleu.metrics import CHRF

PL, OUT = sys.argv[1], sys.argv[2]
REF = Path.home() / ".sacrebleu/wmt23"
N = {"en-de": 557, "zh-en": 400}
hyp = defaultdict(dict)  # (system, pair, run) -> {seg: text}
for f in sorted(glob.glob(f"{PL}/mt_*.jsonl")):
    sysname = os.path.basename(f)[3:-6]
    for line in open(f):
        r = json.loads(line)
        hyp[(sysname, r["pair"], r["run"])][r["seg"]] = r["hyp"]
systems = sorted({k[0] for k in hyp})
runs = sorted({k[2] for k in hyp})
src = {p: (REF / f"wmt23.{p}.src").read_text().splitlines()[:n] for p, n in N.items()}
ref = {p: (REF / f"wmt23.{p}.ref-refA").read_text().splitlines()[:n] for p, n in N.items()}

# Segment-level scores.
chrf = CHRF()
seg = {}
for (s, p, r), h in hyp.items():
    seg[("chrf", s, p, r)] = np.array([chrf.sentence_score(h[i], [ref[p][i]]).score for i in range(N[p])])
try:
    from comet import load_from_checkpoint
    from huggingface_hub import snapshot_download
    model = load_from_checkpoint(snapshot_download("Unbabel/wmt22-comet-da") + "/checkpoints/model.ckpt")
    keys = list(hyp)
    data = [{"src": src[p][i], "mt": hyp[(s, p, r)][i], "ref": ref[p][i]} for (s, p, r) in keys for i in range(N[p])]
    scores = model.predict(data, batch_size=64, gpus=1, progress_bar=False).scores
    k = 0
    for (s, p, r) in keys:
        seg[("comet", s, p, r)] = 100 * np.array(scores[k:k + N[p]])
        k += N[p]
except Exception as e:  # COMET unavailable: report chrF only
    print("COMET skipped:", e)

res = {"systems": systems, "runs": runs, "change": {}, "range": {}, "comparisons": []}
for s, p in itertools.product(systems, N):
    base = hyp[(s, p, "orig_all")]
    changed = [np.mean([hyp[(s, p, r)][i] != base[i] for i in range(N[p])]) for r in runs if r != "orig_all"]
    res["change"][f"{s}|{p}"] = {"mean": float(np.mean(changed)), "max": float(np.max(changed))}
    for m in ("chrf", "comet"):
        if (m, s, p, "orig_all") in seg:
            means = [float(seg[(m, s, p, r)].mean()) for r in runs]
            res["range"][f"{m}|{s}|{p}"] = {"min": min(means), "max": max(means), "range": max(means) - min(means)}

rng = np.random.default_rng(0)
for m, p, (s1, s2) in itertools.product(("chrf", "comet"), N, itertools.combinations(systems, 2)):
    if (m, s1, p, "orig_all") not in seg:
        continue
    idx = rng.integers(0, N[p], size=(1000, N[p]))
    per_run = []
    for r in runs:
        d = seg[(m, s1, p, r)] - seg[(m, s2, p, r)]
        boot = d[idx].mean(1)
        pval = float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))
        per_run.append({"run": r, "delta": float(d.mean()), "p": pval, "sig": pval < 0.05})
    signs = {np.sign(x["delta"]) for x in per_run}
    sigs = {x["sig"] for x in per_run}
    res["comparisons"].append({"metric": m, "pair": p, "a": s1, "b": s2, "runs": per_run,
                               "sign_flip": len(signs) > 1, "sig_flip": len(sigs) > 1})
flips = [c for c in res["comparisons"] if c["sign_flip"] or c["sig_flip"]]
max_comet_range = max((v["range"] for k, v in res["range"].items() if k.startswith("comet|")), default=None)
res["go"] = bool(flips) or (max_comet_range is not None and max_comet_range >= 0.5)
json.dump(res, open(OUT, "w"), indent=1)
print("segments changed vs first run (mean over runs):")
for k, v in res["change"].items():
    print(f"  {k:20s} {v['mean']:.1%} (max {v['max']:.1%})")
print("metric ranges across runs:")
for k, v in res["range"].items():
    print(f"  {k:28s} {v['min']:.2f}-{v['max']:.2f} (range {v['range']:.2f})")
print(f"comparisons: {len(res['comparisons'])}, with sign or significance flips: {len(flips)}")
for c in flips:
    print("  FLIP", c["metric"], c["pair"], c["a"], "vs", c["b"], [(x["run"], round(x["delta"], 2), round(x["p"], 3)) for x in c["runs"]])
print("GO" if res["go"] else "NO-GO", "(rule in PREREG.md)")
