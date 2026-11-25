#!/usr/bin/env python3
"""Judge-robustness analyses A1-A4 and A6 (PROTOCOL.md "Judge-robustness extension to the MT study"); fixed before the
extension runs. Works on any judge-run directory laid out like results/validation/mtvar/judge/<judge>/bi<b>/<run>.

A1 robust verdict changes (Holm within each run's family of comparisons), tie reorderings reported separately.
A2 cross-condition scenario (system 1 judged under c, system 2 under c').
A3a segment bootstrap reference (C2 r1); A3b non-serving reference runs (--ref name=dir, condition C2, BI off).
A4 effect sizes (mean difference, 95% bootstrap CI) per comparison, C2 r1.
A6 length-normalized translation change rates per direction (--translate dir).
Usage: mtvar_ext_score.py --tag <name> --judge-dir <.../bi*> parent [--ref h200=<dir> --ref tp2=<dir>] [--translate dir] [--merge]
"""
import argparse
import gzip
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ["gemma2_9b", "gemma3_12b", "aya_8b", "qwen25_7b", "llama31_8b"]
LPS = ["de_DE", "ja_JP", "is_IS", "sw_KE"]
ALL_LPS = ["de_DE", "zh_CN", "ja_JP", "is_IS", "sw_KE", "gu_IN", "zu_ZA"]
CONDS, REPS = ["C1", "C2", "C3", "C4"], [1, 2]
ALPHA, NPERM, NBOOT = 0.05, 1000, 1000


def load(p):
    return [json.loads(x) for x in (gzip.open(p, "rt") if p.suffix == ".gz" else p.open())]


def run_file(d, c, r):
    for q in (d / f"{c}_r{r}.jsonl", d / f"{c}_r{r}.jsonl.gz"):
        if q.exists():
            return q
    return None


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


def perm_p(d):
    d = np.asarray(d, float)
    if d.size not in _SIGNS:
        _SIGNS[d.size] = np.random.default_rng(0).choice([-1.0, 1.0], size=(NPERM, d.size))
    obs = abs(d.mean())
    return (1 + int((np.abs(_SIGNS[d.size] @ d / d.size) >= obs - 1e-12).sum())) / (NPERM + 1)


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps, kind="stable")
    m = len(ps)
    adj = np.minimum(np.maximum.accumulate((m - np.arange(m)) * ps[order]), 1.0)
    out = np.empty(m)
    out[order] = adj
    return out


def sgn(x):
    return int(np.sign(round(float(x), 12)))


def load_judge(d):
    """scores[(c, r)] = {(sys, lp): np.array over segments in segment order}."""
    out = {}
    for c, r in itertools.product(CONDS, REPS):
        f = run_file(d, c, r)
        if f is None:
            continue
        recs = load(f)
        by = {}
        for x in recs:
            s, rest = x["id"].split("|")
            lp, seg = rest.split(":")
            by.setdefault((s, lp), []).append((int(seg), mqm(x["text"])))
        out[c, r] = {k: np.array([v for _, v in sorted(vs)]) for k, vs in by.items()}
    return out


PAIRS = [(lp, a, b) for lp in LPS for a, b in itertools.combinations(SYSTEMS, 2)]


def family_verdicts(get1, get2):
    """Holm verdicts over the 40 comparisons; get1/get2(lp, sys) -> score arrays."""
    diffs = [get1(lp, a) - get2(lp, b) for lp, a, b in PAIRS]
    ps = [perm_p(d) for d in diffs]
    adj = holm(ps)
    return {k: (sgn(d.mean()), bool(q < ALPHA)) for k, d, q in zip(PAIRS, diffs, adj)}


ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--judge-dir", required=True, help="directory holding bi0/ (and bi1/) of one judge configuration")
ap.add_argument("--ref", action="append", default=[], help="name=dir of a non-serving reference (C2 runs, BI off)")
ap.add_argument("--translate", help="translate run dir (for A6)")
ap.add_argument("--merge", action="store_true")
a = ap.parse_args()
P = f"mtx.{a.tag}"
S = {}
J = load_judge(Path(a.judge_dir) / "bi0")
missing = [k for k in itertools.product(CONDS, REPS) if k not in J]
if missing:
    sys.exit(f"missing runs {missing}")

# A1
V = {k: family_verdicts(lambda lp, s, k=k: J[k][s, lp], lambda lp, s, k=k: J[k][s, lp]) for k in J}
robust, ties, rep_unstable = [], [], 0
for key in PAIRS:
    v = {k: V[k][key] for k in J}
    stable = all(v[c, 1] == v[c, 2] for c in CONDS)
    rep_unstable += not stable
    if not stable:
        continue
    vc = {c: v[c, 1] for c in CONDS}
    if len(set(vc.values())) > 1:
        if any(sig for _, sig in vc.values()):
            robust.append((key, vc))
        else:
            ties.append((key, vc))
S[f"{P}.a1_n"] = len(PAIRS)
S[f"{P}.a1_robust"] = len(robust)
S[f"{P}.a1_tie_reorderings"] = len(ties)
S[f"{P}.a1_repeat_unstable"] = rep_unstable
S[f"{P}.a1_holm_significant_c2"] = sum(sig for _, sig in V["C2", 1].values())
print(f"[{a.tag}] A1: {len(robust)} robust verdict changes, {len(ties)} tie reorderings, {rep_unstable} repeat-unstable, "
      f"of {len(PAIRS)}; Holm-significant in C2 r1: {S[f'{P}.a1_holm_significant_c2']}")
for key, vc in robust:
    print("   robust:", key, vc)

# A2 cross-condition scenario (first repeats)
n_combo = n_diff = n_sign = 0
for c1, c2 in itertools.permutations(CONDS, 2):
    vx = family_verdicts(lambda lp, s: J[c1, 1][s, lp], lambda lp, s: J[c2, 1][s, lp])
    for key in PAIRS:
        n_combo += 1
        same = {V[c1, 1][key], V[c2, 1][key]}
        n_diff += vx[key] not in same
        n_sign += vx[key][0] not in {x[0] for x in same}
S[f"{P}.a2_n"], S[f"{P}.a2_differ"], S[f"{P}.a2_sign_differ"] = n_combo, n_diff, n_sign
print(f"[{a.tag}] A2 cross-condition: {n_diff} of {n_combo} (pair, c, c') verdicts differ from both same-condition "
      f"verdicts ({n_sign} by sign)")

# A3a segment bootstrap of C2 r1
base = V["C2", 1]
rng = np.random.default_rng(0)
n_seg = len(next(iter(J["C2", 1].values())))
chg = chg_sign = 0
for _ in range(NBOOT):
    idx = rng.integers(0, n_seg, n_seg)
    vb = family_verdicts(lambda lp, s: J["C2", 1][s, lp][idx], lambda lp, s: J["C2", 1][s, lp][idx])
    chg += sum(vb[k] != base[k] for k in PAIRS)
    chg_sign += sum(vb[k][0] != base[k][0] for k in PAIRS)
S[f"{P}.a3_boot_verdict_change_rate"] = chg / (NBOOT * len(PAIRS))
S[f"{P}.a3_boot_sign_change_rate"] = chg_sign / (NBOOT * len(PAIRS))
serv = sum(V[c, 1][k] != base[k] for c in CONDS if c != "C2" for k in PAIRS) / (3 * len(PAIRS))
serv_sign = sum(V[c, 1][k][0] != base[k][0] for c in CONDS if c != "C2" for k in PAIRS) / (3 * len(PAIRS))
S[f"{P}.a3_serving_verdict_change_rate"], S[f"{P}.a3_serving_sign_change_rate"] = serv, serv_sign
print(f"[{a.tag}] A3a: verdict change vs C2 r1: serving conditions {serv:.1%} (sign {serv_sign:.1%}); "
      f"segment bootstrap {S[f'{P}.a3_boot_verdict_change_rate']:.1%} (sign {S[f'{P}.a3_boot_sign_change_rate']:.1%})")

# A3b non-serving references
for spec in a.ref:
    name, d = spec.split("=", 1)
    R = load_judge(Path(d) / "bi0")
    for r in REPS:
        if ("C2", r) not in R:
            sys.exit(f"reference {name} missing C2 r{r}")
    vr = family_verdicts(lambda lp, s: R["C2", 1][s, lp], lambda lp, s: R["C2", 1][s, lp])
    S[f"{P}.a3_ref_{name}_verdict_changes"] = sum(vr[k] != base[k] for k in PAIRS)
    S[f"{P}.a3_ref_{name}_sign_changes"] = sum(vr[k][0] != base[k][0] for k in PAIRS)
    items = sum(len(v) for v in R["C2", 1].values())
    S[f"{P}.a3_ref_{name}_score_change"] = sum(int((R["C2", 1][k] != J["C2", 1][k]).sum()) for k in R["C2", 1]) / items
    print(f"[{a.tag}] A3b {name}: {S[f'{P}.a3_ref_{name}_verdict_changes']} verdict changes "
          f"({S[f'{P}.a3_ref_{name}_sign_changes']} sign) vs H100 TP1 C2 r1; item scores changed "
          f"{S[f'{P}.a3_ref_{name}_score_change']:.1%}")

# A4 effect sizes (C2 r1)
rows = []
for lp, s1, s2 in PAIRS:
    d = J["C2", 1][s1, lp] - J["C2", 1][s2, lp]
    bs = [d[rng.integers(0, d.size, d.size)].mean() for _ in range(NBOOT)]
    rows.append({"lp": lp, "a": s1, "b": s2, "diff": float(d.mean()), "lo": float(np.percentile(bs, 2.5)),
                 "hi": float(np.percentile(bs, 97.5)), "p": perm_p(d),
                 "robust": any(k == (lp, s1, s2) for k, _ in robust),
                 "tie_reordering": any(k == (lp, s1, s2) for k, _ in ties)})
out = ROOT / "results" / "validation" / "mtvar" / f"ext_{a.tag}_effects.json"
out.write_text(json.dumps(rows, indent=1))
S[f"{P}.a4_median_abs_diff"] = float(np.median([abs(x["diff"]) for x in rows]))
S[f"{P}.n_items_per_cell"] = n_seg
print(f"[{a.tag}] A4: wrote {out.name}; median |diff| {S[f'{P}.a4_median_abs_diff']:.2f} MQM points, n={n_seg} per cell")

# A6 length-normalized change rates (translations)
if a.translate:
    T = Path(a.translate)
    for lp in ALL_LPS:
        rates, per_tok = [], []
        for s in SYSTEMS:
            runs = {(c, r): {x["id"]: x for x in load(run_file(T / s / "bi0", c, r)) if x["lp"] == lp}
                    for c in CONDS for r in REPS}
            for (c1, r1), (c2, r2) in itertools.combinations(runs, 2):
                if c1 == c2:
                    continue
                A, B = runs[c1, r1], runs[c2, r2]
                diff = [i for i in A if A[i].get("token_sha", A[i].get("text")) != B[i].get("token_sha", B[i].get("text"))]
                toks = np.mean([(A[i]["n_tokens"] + B[i]["n_tokens"]) / 2 for i in A])
                rates.append(len(diff) / len(A))
                per_tok.append(len(diff) / len(A) / toks * 100)
        S[f"{P}.a6.{lp}.change_rate"] = float(np.mean(rates))
        S[f"{P}.a6.{lp}.change_per_100_tokens"] = float(np.mean(per_tok))
        print(f"[{a.tag}] A6 {lp}: change {np.mean(rates):.1%}, per 100 output tokens {np.mean(per_tok):.2%}")
    hi = [S[f"{P}.a6.{lp}.change_per_100_tokens"] for lp in ALL_LPS[:3]]
    lo = [S[f"{P}.a6.{lp}.change_per_100_tokens"] for lp in ALL_LPS[3:]]
    S[f"{P}.a6_high_per_100_tokens"], S[f"{P}.a6_low_per_100_tokens"] = float(np.mean(hi)), float(np.mean(lo))

if a.merge:
    sys.path.insert(0, str(ROOT / "src"))
    from analyze import merge
    merge(S)
    print("merged", len(S), "keys")
