#!/usr/bin/env python3
"""Aggregate statistics for the paper, computed from per-run keys in results/stats.json and merged back as paper.*."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import STATS, merge

S = json.loads(STATS.read_text())
P = {}

# Phase 1: bitwise.
tags = sorted({k[len("phase1."):-len(".bi.bad")] for k in S if k.startswith("phase1.") and k.endswith(".bi.bad")})
h100 = [t for t in tags if t.startswith("h100_") and "stress" not in t]
b200 = [t for t in tags if t.startswith("b200_")]
P["paper.p1_builds_h100"] = len(h100)
P["paper.p1_builds_b200"] = len(b200)
P["paper.p1_rows_per_mode"] = S[f"phase1.{tags[0]}.bi.rows"]
assert all(S[f"phase1.{t}.bi.rows"] == P["paper.p1_rows_per_mode"] for t in tags)
P["paper.p1_bi_bad_total"] = sum(S[f"phase1.{t}.{m}.bad"] for t in tags for m in ("bi", "bi_noenv"))
P["paper.p1_comparisons_total"] = sum(S[f"phase1.{t}.{m}.rows"] for t in tags for m in ("bi", "bi_noenv", "default"))
for gpu, ts in (("h100", h100), ("b200", b200)):
    ctl = [S[f"phase1.{t}.default.bad"] for t in ts]
    P[f"paper.p1_ctl_min_{gpu}"], P[f"paper.p1_ctl_max_{gpu}"] = min(ctl), max(ctl)
cublas = sorted({S[f"phase1.{t}.cublaslt"] for t in tags})
P["paper.p1_cublas_versions"] = len(cublas)

# Phase 2: determinism.
DET = ["q25tp2", "q25tp4", "q25long", "q25samp", "q32tp2", "mistralv2", "gemma2v2"]
for c in DET:
    k = f"phase2.cross.{c}_plain"
    assert S[f"{k}.done"], c
P["paper.p2_pass_configs"] = sum(S[f"phase2.cross.{c}_plain.mismatch_pairs"] == 0 for c in DET)
P["paper.p2_configs"] = len(DET)
# Controls for the configurations in the determinism table only (diagnostic controls excluded).
CTL = ["q25tp2", "q25long", "q25samp", "q32tp2", "mistral", "gemma2"]
ctl = [f"phase2.cross.{c}_off.mismatch_requests" for c in CTL]
fr = [S[k] / S[k.replace("mismatch_requests", "requests")] for k in ctl]
P["paper.p2_ctl_div_min"], P["paper.p2_ctl_div_max"] = min(fr), max(fr)

# Phase 2: cost.
cells = [k[:-len(".plain_loss_median")] for k in S if k.startswith("phase2.") and k.endswith(".plain_loss_median")]
P["paper.p2_cost_cells"] = len(cells)
P["paper.p2_cost_pass"] = sum(S[c + ".plain_loss_median"] <= 0.05 for c in cells)
pre = [c for c in cells if c.startswith("phase2.prefill.")]
P["paper.p2_prefill_plain_max"] = max(S[c + ".plain_loss_median"] for c in pre)
P["paper.p2_prefill_stock_min"] = min(S[c + ".on_loss_median"] for c in pre)
P["paper.p2_prefill_stock_max"] = max(S[c + ".on_loss_median"] for c in pre)

merge(P)
print(len(P), "paper stats")
for k, v in sorted(P.items()):
    print(" ", k, v)
