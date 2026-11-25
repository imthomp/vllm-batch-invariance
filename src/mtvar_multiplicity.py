#!/usr/bin/env python3
"""Post-hoc multiple-testing sensitivity for the preregistered MT judge comparisons.

This does not replace the preregistered per-comparison analysis in mtvar_score.py.
It reapplies Holm and Benjamini-Hochberg adjustment to the same fixed-seed paired
sign-flip permutation p-values, then tests whether the reported condition-sensitive
classification changes. Results merge into results/stats.json and are also written
as a focused sidecar for auditability.
"""
import gzip
import itertools
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/validation/mtvar/judge/qwen25_14b"
STATS = ROOT / "results/stats.json"
OUT = ROOT / "results/validation/mtvar/multiplicity_sensitivity.json"
SYSTEMS = ["gemma2_9b", "gemma3_12b", "aya_8b", "qwen25_7b", "llama31_8b"]
LANGUAGES = ["de_DE", "ja_JP", "is_IS", "sw_KE"]
CONDITIONS = ["C1", "C2", "C3", "C4"]
REPEATS = [1, 2]
ALPHA, NPERM = 0.05, 1000


def load(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return [json.loads(line) for line in f]


def mqm(text):
    n = {"critical": 0, "major": 0, "minor": 0}
    for line in (text or "").lower().splitlines():
        line = re.sub(r"^[\s\-\*•\d\.\)]+", "", line)
        for sev in n:
            if line.startswith(sev):
                n[sev] += 1
                break
    return -(25 * n["critical"] + 5 * n["major"] + n["minor"])


_SIGNS = {}


def test(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    if d.size not in _SIGNS:
        _SIGNS[d.size] = np.random.default_rng(0).choice([-1.0, 1.0], size=(NPERM, d.size))
    obs = abs(d.mean())
    p = (1 + int((np.abs((_SIGNS[d.size] * d).mean(axis=1)) >= obs - 1e-12).sum())) / (NPERM + 1)
    return int(np.sign(round(d.mean(), 12))), p


def adjust(pairs, method):
    """Return adjusted p-values by original key, using standard step-down/step-up rules."""
    keys = list(pairs)
    p = np.asarray([pairs[k] for k in keys], dtype=float)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    m = len(p)
    if method == "holm":
        adj_ranked = np.maximum.accumulate((m - np.arange(m)) * ranked)
    elif method == "bh":
        adj_ranked = np.minimum.accumulate((m * ranked / np.arange(1, m + 1))[::-1])[::-1]
    else:
        raise ValueError(method)
    adj_ranked = np.minimum(adj_ranked, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = adj_ranked
    return dict(zip(keys, out.tolist()))


def sensitive(v):
    rep_stable = all(v[c, 1] == v[c, 2] for c in CONDITIONS)
    return rep_stable and len({v[c, 1] for c in CONDITIONS}) > 1


# Load the eight archived judge runs and preserve their common item order.
runs = {}
for c, r in itertools.product(CONDITIONS, REPEATS):
    path = next((p for p in (BASE / "bi0" / f"{c}_r{r}.jsonl", BASE / "bi0" / f"{c}_r{r}.jsonl.gz") if p.exists()), None)
    meta = BASE / "bi0" / f"{c}_r{r}.meta.json"
    assert path and meta.is_file(), f"missing judge run {c}_r{r}"
    assert not json.loads(meta.read_text())["limit"], f"pilot data: {path}"
    runs[c, r] = load(path)
ids = [r["id"] for r in runs["C2", 1]]
assert all([r["id"] for r in rows] == ids for rows in runs.values())

systems_of = [x.split("|")[0] for x in ids]
languages_of = [x.split("|")[1].split(":")[0] for x in ids]
scores = {(c, r): np.asarray([mqm(x["text"]) for x in rows], dtype=float) for (c, r), rows in runs.items()}
comparisons = []
pvals = {}
for lp in LANGUAGES:
    for s1, s2 in itertools.combinations(SYSTEMS, 2):
        i1 = [i for i, (s, l) in enumerate(zip(systems_of, languages_of)) if s == s1 and l == lp]
        i2 = [i for i, (s, l) in enumerate(zip(systems_of, languages_of)) if s == s2 and l == lp]
        assert [ids[i].split("|")[1] for i in i1] == [ids[i].split("|")[1] for i in i2]
        comp = (lp, s1, s2)
        comparisons.append(comp)
        for c, r in itertools.product(CONDITIONS, REPEATS):
            sign, p = test(scores[c, r][i1], scores[c, r][i2])
            pvals[comp, c, r] = (sign, p)
assert len(comparisons) == 40

# Adjustment family 1: all system-pair/direction comparisons within each of the eight runs.
run_adjusted = {method: {} for method in ("holm", "bh")}
for method in run_adjusted:
    for c, r in itertools.product(CONDITIONS, REPEATS):
        fam = {comp: pvals[comp, c, r][1] for comp in comparisons}
        run_adjusted[method].update({(comp, c, r): p for comp, p in adjust(fam, method).items()})

# Adjustment family 2: all comparison outcomes across the complete 8-run design.
global_adjusted = {method: adjust({(comp, c, r): pvals[comp, c, r][1]
                                   for comp, c, r in itertools.product(comparisons, CONDITIONS, REPEATS)}, method)
                   for method in ("holm", "bh")}

summary = {
    "method": "post-hoc fixed-seed paired sign-flip sensitivity; per-run families of 40 and one global family of 320",
    "comparison_count": len(comparisons),
    "condition_repeat_run_count": len(CONDITIONS) * len(REPEATS),
}
for name, corrected in (("unadjusted", None), ("holm_per_run", run_adjusted["holm"]),
                        ("bh_per_run", run_adjusted["bh"]), ("holm_global", global_adjusted["holm"]),
                        ("bh_global", global_adjusted["bh"])):
    n = 0
    sign_n = 0
    sig_n = 0
    details = []
    for comp in comparisons:
        v = {}
        for c, r in itertools.product(CONDITIONS, REPEATS):
            sign, p = pvals[comp, c, r]
            if corrected is None:
                is_sig = p < ALPHA
            elif name.endswith("global"):
                is_sig = corrected[comp, c, r] < ALPHA
            else:
                is_sig = corrected[comp, c, r] < ALPHA
            v[c, r] = (sign, is_sig)
        if sensitive(v):
            n += 1
            sign_change = len({v[c, 1][0] for c in CONDITIONS}) > 1
            sig_change = len({v[c, 1][1] for c in CONDITIONS}) > 1
            sign_n += sign_change
            sig_n += sig_change
            details.append({"language": comp[0], "system_a": comp[1], "system_b": comp[2],
                            "sign_changed": sign_change, "significance_changed": sig_change})
    summary[f"{name}.condition_sensitive"] = n
    summary[f"{name}.sign_changed"] = sign_n
    summary[f"{name}.significance_changed"] = sig_n
    summary[f"{name}.details"] = details

# The unadjusted result must reproduce the registered scorer before this sensitivity is trusted.
stored = json.loads(STATS.read_text())
assert summary["unadjusted.condition_sensitive"] == stored["mtvar.judge_comparisons_cond"] == 5
assert summary["unadjusted.sign_changed"] == stored["mtvar.judge_comparisons_cond_sign"] == 3

# Merge review-only aggregates into the canonical stats source; never replace its existing keys.
new_stats = dict(stored)
for key in ("condition_sensitive", "sign_changed", "significance_changed"):
    for method in ("unadjusted", "holm_per_run", "bh_per_run", "holm_global", "bh_global"):
        new_stats[f"mtvar.multiplicity.{method}.{key}"] = summary[f"{method}.{key}"]
STATS.write_text(json.dumps(new_stats, indent=2, sort_keys=True) + "\n")
OUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
for method in ("unadjusted", "holm_per_run", "bh_per_run", "holm_global", "bh_global"):
    print(method, summary[f"{method}.condition_sensitive"], "condition-sensitive comparisons;",
          summary[f"{method}.significance_changed"], "with significance changes;",
          summary[f"{method}.sign_changed"], "with score-order changes")
