#!/usr/bin/env python3
"""Score the MT eval-variance study (PROTOCOL.md "Do serving conditions change MT evaluation conclusions?").
Fixed before any full run; later changes are recorded in PROTOCOL.md with their reason.

Inputs: <root>/translate/<sys>/bi<b>/<cond>_r<rep>.jsonl[.gz], <root>/judge/qwen25_14b/bi<b>/..., and the metric scores
from src/mtvar_metrics.py (--metrics). Verdict per comparison and run: (sign of the mean segment-score difference,
paired sign-flip permutation p < 0.05 with 1,000 permutations and a fixed seed).
Usage: mtvar_score.py --root <dir> --metrics <scores.jsonl.gz> [--merge] [--partial]
"""
import argparse
import gzip
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ["gemma2_9b", "gemma3_12b", "aya_8b", "qwen25_7b", "llama31_8b"]
JUDGE = "qwen25_14b"
LPS = ["de_DE", "zh_CN", "ja_JP", "is_IS", "sw_KE", "gu_IN", "zu_ZA"]
JUDGE_LPS = ["de_DE", "ja_JP", "is_IS", "sw_KE"]
METRICS = {"metricx": LPS, "comet22": [x for x in LPS if x != "zu_ZA"], "kiwi": [x for x in LPS if x != "zu_ZA"]}
CONDS, REPS = ["C1", "C2", "C3", "C4"], [1, 2]
ALPHA, NPERM = 0.05, 1000


def load(p):
    return [json.loads(x) for x in (gzip.open(p, "rt") if p.suffix == ".gz" else p.open())]


def tok_key(r):
    return r["token_sha"] if "token_sha" in r else hashlib.sha256(json.dumps(r["token_ids"]).encode()).hexdigest()


def mqm(text):
    """GEMBA-MQM-style score from the judge output: -(25 critical + 5 major + 1 minor); no category named -> 0."""
    n = {"critical": 0, "major": 0, "minor": 0}
    for line in (text or "").lower().splitlines():
        line = re.sub(r"^[\s\-\*•\d\.\)]+", "", line)
        for sev in n:
            if line.startswith(sev):
                n[sev] += 1
                break
    return -(25 * n["critical"] + 5 * n["major"] + n["minor"])


_SIGNS = {}


def verdict(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    if d.size not in _SIGNS:
        _SIGNS[d.size] = np.random.default_rng(0).choice([-1.0, 1.0], size=(NPERM, d.size))
    obs = abs(d.mean())
    p = (1 + int((np.abs((_SIGNS[d.size] * d).mean(axis=1)) >= obs - 1e-12).sum())) / (NPERM + 1)
    return (int(np.sign(round(d.mean(), 12))), bool(p < ALPHA))


def load_runs(base, names, partial):
    runs = {}
    for s, bi, c, r in itertools.product(names, (0, 1), CONDS, REPS):
        d = base / s / f"bi{bi}"
        p = next((q for q in (d / f"{c}_r{r}.jsonl", d / f"{c}_r{r}.jsonl.gz") if q.exists()), None)
        m = d / f"{c}_r{r}.meta.json"
        if p is None or not m.exists():
            if partial:
                continue
            sys.exit(f"missing {d}/{c}_r{r}")
        meta = json.loads(m.read_text())
        if meta["limit"]:
            sys.exit(f"{p} is a pilot run (limit set)")
        runs[s, bi, c, r] = (load(p), meta)
    return runs


def classify(v):
    """v[(c, r)] -> verdict. Returns (differs between any runs, differs between repeats, condition-sensitive)."""
    rep_stable = all(v[c, 1] == v[c, 2] for c in CONDS)
    return len(set(v.values())) > 1, not rep_stable, rep_stable and len({v[c, 1] for c in CONDS}) > 1


ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--metrics", required=True)
ap.add_argument("--merge", action="store_true")
ap.add_argument("--partial", action="store_true")
a = ap.parse_args()
R = Path(a.root)
S, P = {}, "mtvar"

# ---- translations
tr = load_runs(R / "translate", SYSTEMS, a.partial)
ids = None
for k, (recs, _) in tr.items():
    if ids is None:
        ids = [x["id"] for x in recs]
    assert [x["id"] for x in recs] == ids, k
assert ids and len(ids) == 960 * len(LPS), len(ids or [])
lp_of = [i.split(":")[0] for i in ids]
idx = {lp: [i for i, x in enumerate(lp_of) if x == lp] for lp in LPS}
sc = {}
for x in load(Path(a.metrics)):
    sc[x["id"], x["hyp"]] = x
seg = {}  # seg[k][metric] = np.array over all ids
for k, (recs, _) in tr.items():
    rows = [sc[r["id"], r["text"].strip()] for r in recs]
    seg[k] = {m: np.array([np.nan if z[m] is None else z[m] for z in rows]) for m in METRICS}
systems = [s for s in SYSTEMS if all((s, 0, c, r) in tr for c in CONDS for r in REPS)]
print("systems with all 8 off runs:", systems)

for s in systems:
    for lp in LPS:
        I = idx[lp]
        txt = {(c, r): [tr[s, 0, c, r][0][i]["text"].strip() for i in I] for c in CONDS for r in REPS}

        def ch(k1, k2):
            return sum(x != y for x, y in zip(txt[k1], txt[k2])) / len(I)
        between = [ch((c1, r1), (c2, r2)) for c1, c2 in itertools.combinations(CONDS, 2) for r1 in REPS for r2 in REPS]
        within = [ch((c, 1), (c, 2)) for c in CONDS]
        k = f"{P}.{s}.{lp}"
        S[k + ".change_between_mean"], S[k + ".change_between_max"] = float(np.mean(between)), max(between)
        S[k + ".change_within_mean"], S[k + ".change_within_max"] = float(np.mean(within)), max(within)
        for m, lps in METRICS.items():
            if lp in lps:
                v = [float(np.mean(seg[s, 0, c, r][m][I])) for c in CONDS for r in REPS]
                S[f"{k}.{m}_spread"] = max(v) - min(v)
    print(f"{s:11s} segment change between conditions (mean over directions) "
          f"{np.mean([S[f'{P}.{s}.{lp}.change_between_mean'] for lp in LPS]):.2%}, within "
          f"{np.mean([S[f'{P}.{s}.{lp}.change_within_mean'] for lp in LPS]):.2%}")

tot = {"any": 0, "repeat": 0, "cond": 0, "n": 0, "cond_sign": 0}
detail, ranks = [], {}
for m, lps in METRICS.items():
    for lp in lps:
        I = idx[lp]
        rk = set()
        for c in CONDS:
            for r in REPS:
                means = {s: np.mean(seg[s, 0, c, r][m][I]) for s in systems}
                rk.add(tuple(sorted(systems, key=lambda s: -means[s])))
        ranks[m, lp] = len(rk)
        S[f"{P}.{m}.{lp}.distinct_rankings"] = len(rk)
        for s1, s2 in itertools.combinations(systems, 2):
            v = {(c, r): verdict(seg[s1, 0, c, r][m][I], seg[s2, 0, c, r][m][I]) for c in CONDS for r in REPS}
            any_, rep, cond = classify(v)
            tot["n"] += 1
            tot["any"] += any_
            tot["repeat"] += rep
            tot["cond"] += cond
            if cond:
                detail.append(f"{m} {lp} {s1} vs {s2}: " + ", ".join(f"{c} {v[c, 1]}" for c in CONDS))
                tot["cond_sign"] = tot.get("cond_sign", 0) + (len({v[c, 1][0] for c in CONDS}) > 1)
for kk, vv in tot.items():
    S[f"{P}.sys_comparisons_{kk}"] = vv
S[f"{P}.sys_rankings_changed"] = sum(v > 1 for v in ranks.values())
print(f"system comparisons {tot['n']}: differ between some runs {tot['any']}, between repeats {tot['repeat']}, "
      f"condition-sensitive {tot['cond']}; direction x metric cells with >1 ranking: {S[f'{P}.sys_rankings_changed']}")
for d in detail:
    print("  ", d)

# ---- judge
jr = load_runs(R / "judge", [JUDGE], a.partial)
if all((JUDGE, 0, c, r) in jr for c in CONDS for r in REPS):
    jids = [x["id"] for x in jr[JUDGE, 0, "C2", 1][0]]
    for k, (recs, _) in jr.items():
        assert [x["id"] for x in recs] == jids, k
    jsys = [i.split("|")[0] for i in jids]
    jlp = [i.split("|")[1].split(":")[0] for i in jids]
    js = {k: np.array([mqm(x["text"]) for x in recs]) for k, (recs, _) in jr.items()}
    jt = {"any": 0, "repeat": 0, "cond": 0, "n": 0, "cond_sign": 0}
    jd = []
    for lp in JUDGE_LPS:
        for s1, s2 in itertools.combinations(SYSTEMS, 2):
            I1 = [i for i in range(len(jids)) if jlp[i] == lp and jsys[i] == s1]
            I2 = [i for i in range(len(jids)) if jlp[i] == lp and jsys[i] == s2]
            assert [jids[i].split("|")[1] for i in I1] == [jids[i].split("|")[1] for i in I2]
            v = {(c, r): verdict(js[JUDGE, 0, c, r][I1], js[JUDGE, 0, c, r][I2]) for c in CONDS for r in REPS}
            any_, rep, cond = classify(v)
            jt["n"] += 1
            jt["any"] += any_
            jt["repeat"] += rep
            jt["cond"] += cond
            if cond:
                jd.append(f"judge {lp} {s1} vs {s2}: " + ", ".join(f"{c} {v[c, 1]}" for c in CONDS))
                jt["cond_sign"] = jt.get("cond_sign", 0) + (len({v[c, 1][0] for c in CONDS}) > 1)
    for kk, vv in jt.items():
        S[f"{P}.judge_comparisons_{kk}"] = vv
    jtxt = {k: [x["text"] for x in recs] for k, (recs, _) in jr.items()}
    jb = [np.mean([x != y for x, y in zip(jtxt[JUDGE, 0, c1, r1], jtxt[JUDGE, 0, c2, r2])])
          for c1, c2 in itertools.combinations(CONDS, 2) for r1 in REPS for r2 in REPS]
    jw = [np.mean([x != y for x, y in zip(jtxt[JUDGE, 0, c, 1], jtxt[JUDGE, 0, c, 2])]) for c in CONDS]
    js_b = [np.mean(js[JUDGE, 0, c1, r1] != js[JUDGE, 0, c2, r2])
            for c1, c2 in itertools.combinations(CONDS, 2) for r1 in REPS for r2 in REPS]
    S[f"{P}.judge.change_between_mean"], S[f"{P}.judge.change_within_mean"] = float(np.mean(jb)), float(np.mean(jw))
    S[f"{P}.judge.score_change_between_mean"] = float(np.mean(js_b))
    print(f"judge: output changes between conditions {np.mean(jb):.2%} (within {np.mean(jw):.2%}); score changes "
          f"{np.mean(js_b):.2%}; comparisons {jt['n']}: differ {jt['any']}, between repeats {jt['repeat']}, "
          f"condition-sensitive {jt['cond']}")
    for d in jd:
        print("  ", d)
    allruns = {**{("tr",) + k: v for k, v in tr.items()}, **{("judge",) + k: v for k, v in jr.items()}}
else:
    print("judge runs incomplete; judge not scored")
    allruns = {("tr",) + k: v for k, v in tr.items()}

# ---- batch-invariant mode on: identity and time
bi_pass = True
for name in systems + [JUDGE]:
    kind = "judge" if name == JUDGE else "tr"
    on = [allruns[kind, name, 1, c, r][0] for c in CONDS for r in REPS if (kind, name, 1, c, r) in allruns]
    if len(on) < 8:
        continue
    diff = sum(len({tok_key(run[i]) for run in on}) > 1 for i in range(len(on[0])))
    S[f"{P}.{name}.bi_on_requests_differing"] = diff
    bi_pass &= diff == 0
    for c in CONDS:
        t = {b: np.mean([allruns[kind, name, b, c, r][1]["gen_seconds"] for r in REPS]) for b in (0, 1)
             if all((kind, name, b, c, r) in allruns for r in REPS)}
        if len(t) == 2:
            S[f"{P}.{name}.{c}.time_on_vs_off"] = t[1] / t[0] - 1
    print(f"{name:11s} BI on: {diff} requests differ across 8 runs; time on/off "
          + " ".join(f"{c} {S.get(f'{P}.{name}.{c}.time_on_vs_off', float('nan')):+.0%}" for c in CONDS))
S[f"{P}.bi_on_pass"] = bool(bi_pass)
side = Path(a.metrics).with_name("metric_batch_side.json")
if side.exists():
    for k, v in json.loads(side.read_text()).items():
        if k != "run":
            S[f"{P}.side.{k}"] = v
S[f"{P}.gpus"] = sorted({v[1]["gpu"] for v in allruns.values()})
claim_sys = tot["cond"] > 0
claim_judge = S.get(f"{P}.judge_comparisons_cond", 0) > 0
S[f"{P}.claim_systems"], S[f"{P}.claim_judge"] = claim_sys, claim_judge
print("BI-on arm:", "PASS" if bi_pass else "FAIL (investigate before any claim)")
print("Decision (systems):", "conditions change conclusions" if claim_sys else "no condition-sensitive comparison")
print("Decision (judge):", "conditions change conclusions" if claim_judge else "no condition-sensitive comparison")
if a.merge:
    if a.partial or len(systems) < len(SYSTEMS) or f"{P}.judge_comparisons_n" not in S:
        sys.exit("not merging partial results")
    sys.path.insert(0, str(ROOT / "src"))
    from analyze import merge
    merge(S)
    print("merged", len(S), "keys")
