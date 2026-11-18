#!/usr/bin/env python3
"""Score the eval-variance runs (PROTOCOL.md "Do serving conditions change NLP evaluation conclusions?"). Fixed before
any run; changes after data are recorded in PROTOCOL.md with their reason.

Input: <root>/<model_short>/bi<0|1>/<cond>_r<rep>.jsonl from src/evalvar_gen.py.
Grading: a response is correct iff finish == "stop", it contains \\boxed{...}, and norm(boxed) == norm(gold), with the
same normalizer for every arm. Truncated (finish == "length") responses are wrong and counted separately.
Usage: evalvar_score.py --root <dir> [--merge] [--partial]   (--partial scores whatever exists and never merges)
"""
import argparse
import itertools
import json
import re
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["qwen25_7b", "llama31_8b", "gemma2_9b", "mistral_7b", "olmo2_7b", "phi4_mini"]
CONDS = ["C1", "C2", "C3", "C4"]
REPS = [1, 2]
BENCH = ["gsm8k", "math500"]
ALPHA = 0.05


def boxed(text):
    m = None
    for m in re.finditer(r"\\boxed\{", text or ""):
        pass
    if not m:
        return None
    i, d = m.end(), 1
    while i < len(text) and d:
        d += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[m.end():i - 1]


def norm(s):
    s = re.sub(r"\\(left|right|!|,|;| )", "", s or "").replace("dfrac", "frac").replace("tfrac", "frac")
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s)
    s = s.replace("\\$", "").replace("\\%", "").replace("$", "").replace(" ", "").rstrip(".").replace(",", "")
    try:
        return str(float(s))
    except ValueError:
        return s


def grade(r):
    b = boxed(r["text"]) if r["finish"] == "stop" else None
    return b is not None and norm(b) == norm(r["gold"])


def mcnemar_p(x, y):
    """Exact two-sided McNemar p from paired 0/1 lists."""
    b = sum(1 for u, v in zip(x, y) if u and not v)
    c = sum(1 for u, v in zip(x, y) if v and not u)
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def verdict(x, y):
    """(sign of accuracy difference, significant at ALPHA)."""
    d = sum(x) - sum(y)
    return ((d > 0) - (d < 0), mcnemar_p(x, y) < ALPHA)


ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--merge", action="store_true")
ap.add_argument("--partial", action="store_true")
a = ap.parse_args()
R = Path(a.root)

runs, ids = {}, None  # runs[m, bi, cond, rep] = list of records in problem order
for m, bi, c, r in itertools.product(MODELS, (0, 1), CONDS, REPS):
    p = R / m / f"bi{bi}" / f"{c}_r{r}.jsonl"
    if not p.exists() or not (meta := p.with_suffix(".meta.json")).exists():
        if a.partial:
            continue
        sys.exit(f"missing {p}")
    recs = [json.loads(line) for line in p.open()]
    if json.loads(meta.read_text())["limit"]:
        sys.exit(f"{p} is a smoke run (limit set)")
    if ids is None:
        ids = [x["id"] for x in recs]
    assert [x["id"] for x in recs] == ids, p
    runs[m, bi, c, r] = recs
assert ids and len(ids) == 1819, "no complete runs" if not ids else len(ids)
src = {x["id"]: x["src"] for x in next(iter(runs.values()))}
idx = {b: [i for i, q in enumerate(ids) if src[q] == b] for b in BENCH}
corr = {k: [grade(x) for x in v] for k, v in runs.items()}
meta = {k: json.loads((R / k[0] / f"bi{k[1]}" / f"{k[2]}_r{k[3]}.meta.json").read_text()) for k in runs}

S, P = {}, "evalvar"
models = [m for m in MODELS if all((m, 0, c, r) in runs for c in CONDS for r in REPS)]
print(f"models with all 8 off runs: {models}")

# Per model x benchmark: accuracy, spreads, flip rates (mode off)
for m in models:
    for b in BENCH:
        I = idx[b]
        acc = {(c, r): sum(corr[m, 0, c, r][i] for i in I) / len(I) for c in CONDS for r in REPS}
        trunc = max(sum(runs[m, 0, c, r][i]["finish"] == "length" for i in I) / len(I) for c in CONDS for r in REPS)
        k = f"{P}.{m}.{b}"
        for (c, r), v in acc.items():
            S[f"{k}.acc_{c}_r{r}"] = v
        S[f"{k}.acc_spread_all"] = max(acc.values()) - min(acc.values())
        S[f"{k}.acc_spread_repeat_max"] = max(abs(acc[c, 1] - acc[c, 2]) for c in CONDS)
        S[f"{k}.trunc_max"] = trunc

        def flips(k1, k2):
            return sum(corr[(m, 0) + k1][i] != corr[(m, 0) + k2][i] for i in I) / len(I)
        between = [flips((c1, r1), (c2, r2)) for c1, c2 in itertools.combinations(CONDS, 2) for r1 in REPS for r2 in REPS]
        within = [flips((c, 1), (c, 2)) for c in CONDS]
        S[f"{k}.flip_between_mean"], S[f"{k}.flip_between_max"] = sum(between) / len(between), max(between)
        S[f"{k}.flip_within_mean"], S[f"{k}.flip_within_max"] = sum(within) / len(within), max(within)
        print(f"{m:11s} {b:8s} acc {min(acc.values()):.3f}-{max(acc.values()):.3f} spread {S[k + '.acc_spread_all']:.3f}"
              f" (repeat {S[k + '.acc_spread_repeat_max']:.3f}) flips between {S[k + '.flip_between_mean']:.3%}"
              f" within {S[k + '.flip_within_mean']:.3%} trunc<= {trunc:.1%}")

# Pairwise comparisons per off run; classify each comparison
n_pairs = changed_cond = changed_only_cond = changed_rep = 0
detail = []
for b in BENCH:
    I = idx[b]
    for m1, m2 in itertools.combinations(models, 2):
        n_pairs += 1
        v = {(c, r): verdict([corr[m1, 0, c, r][i] for i in I], [corr[m2, 0, c, r][i] for i in I])
             for c in CONDS for r in REPS}
        across = len({v[c1, r1] for c1 in CONDS for r1 in REPS}) > 1  # differs between some two runs
        rep_stable = all(v[c, 1] == v[c, 2] for c in CONDS)
        cond_diff = len({v[c, 1] for c in CONDS}) > 1
        changed_cond += across
        changed_rep += not rep_stable
        if cond_diff and rep_stable:
            changed_only_cond += 1
            detail.append(f"{b} {m1} vs {m2}: " + ", ".join(f"{c} {v[c, 1]}" for c in CONDS))
S[f"{P}.comparisons"] = n_pairs
S[f"{P}.comparisons_changed_any"] = changed_cond
S[f"{P}.comparisons_changed_repeat"] = changed_rep
S[f"{P}.comparisons_changed_condition_only"] = changed_only_cond
claim = changed_only_cond > 0
S[f"{P}.claim_conditions_change_conclusions"] = claim
print(f"comparisons {n_pairs}: verdict differs between some runs {changed_cond}, between repeats {changed_rep}, "
      f"between conditions with stable repeats {changed_only_cond}")
for d in detail:
    print("  ", d)

# Rankings per off run (by accuracy, ties by model order)
for b in BENCH:
    ranks = set()
    for c in CONDS:
        for r in REPS:
            acc = {m: sum(corr[m, 0, c, r][i] for i in idx[b]) for m in models}
            ranks.add(tuple(sorted(models, key=lambda m: (-acc[m], MODELS.index(m)))))
    S[f"{P}.{b}.distinct_rankings"] = len(ranks)
    print(f"{b}: {len(ranks)} distinct model rankings across the 8 off runs")

# Batch-invariant mode on: identical outputs across all 8 runs; time vs off
bi_pass = True
for m in MODELS:
    on = [runs[m, 1, c, r] for c in CONDS for r in REPS if (m, 1, c, r) in runs]
    if len(on) < 8:
        continue
    diff = sum(len({tuple(run[i]["token_ids"]) for run in on}) > 1 for i in range(len(ids)))
    S[f"{P}.{m}.bi_on_requests_differing"] = diff
    bi_pass &= diff == 0
    for c in CONDS:
        if all((m, bi, c, r) in meta for bi in (0, 1) for r in REPS):
            t = {bi: sorted(meta[m, bi, c, r]["gen_seconds"] for r in REPS) for bi in (0, 1)}
            S[f"{P}.{m}.{c}.time_on_vs_off"] = (sum(t[1]) / 2) / (sum(t[0]) / 2) - 1
    acc_on = {b: sum(corr[m, 1, "C2", 1][i] for i in idx[b]) / len(idx[b]) for b in BENCH}
    for b in BENCH:
        S[f"{P}.{m}.{b}.acc_bi_on"] = acc_on[b]
    print(f"{m:11s} BI on: {diff} requests differ across 8 runs; time on/off "
          + " ".join(f"{c} {S.get(f'{P}.{m}.{c}.time_on_vs_off', float('nan')):+.0%}" for c in CONDS))
S[f"{P}.bi_on_pass"] = bi_pass
S[f"{P}.gpus"] = sorted({v["gpu"] for v in meta.values()})
print("BI-on arm:", "PASS" if bi_pass else "FAIL (investigate before any claim)")
print("Decision:", "serving conditions change evaluation conclusions" if claim else
      "no comparison changed between conditions beyond repeat noise")
if a.merge:
    if a.partial or len(models) < len(MODELS):
        sys.exit("not merging partial results")
    sys.path.insert(0, str(ROOT / "src"))
    from analyze import merge
    merge(S)
    print("merged", len(S), "keys")
