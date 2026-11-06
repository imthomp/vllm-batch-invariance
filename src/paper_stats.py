#!/usr/bin/env python3
"""Aggregate statistics for the paper, computed from per-run keys in results/stats.json and merged back as paper.*."""
import json
import math
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
P["paper.p1_bi_comparisons"] = sum(S[f"phase1.{t}.{m}.rows"] for t in tags for m in ("bi", "bi_noenv"))
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

# Phase 1c: per-shape map at the dense grid (results/phase1c/shape_map_bi.json, from src/shape_map.py).
shape_map = json.loads((STATS.parent / "phase1c" / "shape_map_bi.json").read_text())
fail_n = sorted({int(k.split("x")[0]) for k, v in shape_map.items() if v["bad"]})
pass_n = sorted({int(k.split("x")[0]) for k, v in shape_map.items() if not v["bad"]} - set(fail_n))
P["paper.p1c_shapes"] = len({k.split("|")[0] for k in shape_map})
P["paper.p1c_max_fail_n"] = max(fail_n)
P["paper.p1c_min_pass_n"] = min(pass_n)
P["paper.p1c_router_bf16_first_bad_m"] = shape_map["32x2880|bfloat16"]["first_bad_m"]
P["paper.p1c_router_fp16_first_bad_m"] = shape_map["32x2880|float16"]["first_bad_m"]
assert max(fail_n) < min(pass_n), "N boundary is not clean"

# Phase 1d: finer sweep (results/phase1d, from src/shape_map.py), read together with Phase 1c.
def cells(mode):
    t = {}
    for d in ("phase1c", "phase1d"):
        for key, v in json.loads((STATS.parent / d / f"shape_map_{mode}.json").read_text()).items():
            nk = tuple(int(x) for x in key.split("|")[0].split("x"))
            t[nk] = t.get(nk, False) or bool(v["bad"])  # True if any dtype fails
    return t


bi_c, ne_c = cells("bi"), cells("bi_noenv")
d1 = json.loads((STATS.parent / "phase1d" / "shape_map_bi.json").read_text())
mid = sorted({n for n, k in bi_c if P["paper.p1c_max_fail_n"] < n < P["paper.p1c_min_pass_n"]})
mid_pass = sorted({n for n in mid if not any(f for (n2, k), f in bi_c.items() if n2 == n)})
g = math.gcd(*mid_pass)
assert g > 1 and all((n % g == 0) == (n in mid_pass) for n in mid), "mid-range pass set is not 'multiples of g'"
assert set(range(mid[0], mid[-1] + 1)) == set(mid), "mid-range sweep has gaps"
P["paper.p1d_mid_lo"], P["paper.p1d_mid_hi"], P["paper.p1d_mid_pass_multiple"] = mid[0], mid[-1], g
narrow = sorted({n for n, k in bi_c if n <= P["paper.p1c_max_fail_n"] and len({k2 for n2, k2 in bi_c if n2 == n}) > 2})
nk_fail = sorted({k for (n, k), f in bi_c.items() if n in narrow and f})
nk_pass = sorted({k for (n, k), f in bi_c.items() if n in narrow and not f})
assert all(bi_c[(n, k)] == (k in nk_fail) for n in narrow for k in nk_fail + nk_pass if (n, k) in bi_c), "narrow K split not uniform"
P["paper.p1d_narrow_ns"], P["paper.p1d_narrow_fail_k"], P["paper.p1d_narrow_pass_k"] = narrow, nk_fail, nk_pass
# Without the workspace variables (flags only).
ne_fail_k = sorted({k for (n, k), f in ne_c.items() if f and n >= P["paper.p1c_min_pass_n"]})
ne_max_fail_n = max(n for (n, k), f in ne_c.items() if f)
ne_min_pass_n = min(n for (n, k) in ne_c if n > ne_max_fail_n)
assert all(f for (n, k), f in ne_c.items() if n <= ne_max_fail_n and k in ne_fail_k), "flags-only: some width up to the max passes at a failing K"
assert not any(f for (n, k), f in ne_c.items() if n >= P["paper.p1c_min_pass_n"] and k not in ne_fail_k), "flags-only: a wide shape fails outside the failing K set"
assert not any(f for (n, k), f in ne_c.items() if n >= ne_min_pass_n), "flags-only: a wide shape fails"
P["paper.p1d_noenv_fail_k"], P["paper.p1d_noenv_max_fail_n"], P["paper.p1d_noenv_min_pass_n"] = ne_fail_k, ne_max_fail_n, ne_min_pass_n
P["paper.p1d_noenv_min_fail_n"] = min(n for (n, k), f in ne_c.items() if f and k in ne_fail_k and n >= min(narrow))

# Phase 1d on H200 (results/phase1d_h200): per-shape agreement with H100, both modes.
def cells_dir(d, mode):
    t = {}
    for key, v in json.loads((STATS.parent / d / f"shape_map_{mode}.json").read_text()).items():
        nk = tuple(int(x) for x in key.split("|")[0].split("x"))
        t[nk] = t.get(nk, False) or bool(v["bad"])
    return t


for mode in ("bi", "bi_noenv"):
    h1, h2 = cells_dir("phase1d", mode), cells_dir("phase1d_h200", mode)
    assert set(h1) == set(h2), "H100 and H200 Phase 1d shape sets differ"
    P[f"paper.p1d_h200_{mode}_shapes"] = len(h2)
    P[f"paper.p1d_h200_{mode}_agree"] = sum(h1[k] == h2[k] for k in h2)

# gpt-oss-20b, vLLM bs1_vs_bsN suite (results/prv2/gptoss_hum_*.log.gz). FLEX_ATTENTION (no attention sinks) and
# the online-FP8 test (model not cached) are excluded per PROTOCOL.md; they fail identically in every arm.
import gzip as _gz
import re as _re
for arm in ("on", "blanket", "v2"):
    text = _gz.open(STATS.parent / "prv2" / f"gptoss_hum_{arm}.log.gz", "rt").read()
    res = dict((t, st_) for st_, t in _re.findall(r"^(PASSED|FAILED) test_batch_invariance\.py::(\S+)", text, _re.M))
    counted = {t: v for t, v in res.items() if "FLEX_ATTENTION" not in t and "online_fp8" not in t}
    P[f"paper.gptoss_{arm}_tests"] = len(counted)
    P[f"paper.gptoss_{arm}_pass"] = sum(v == "PASSED" for v in counted.values())
assert P["paper.gptoss_on_tests"] == P["paper.gptoss_v2_tests"] == P["paper.gptoss_blanket_tests"]

# vLLM #65627 (FlashAttention sliding-window fix), issue reproducer on H100 (results/pr65627/v65627r_*.out).
rep = _re.findall(r"RESULT (\d+)/(\d+) prompts differ", open(next((STATS.parent / "pr65627").glob("v65627r_*.out"))).read())
assert len(rep) == 2, "expected base and fix arms"
(P["paper.fafix_base_div"], P["paper.fafix_n"]), (P["paper.fafix_fix_div"], _n2) = [(int(a), int(b)) for a, b in rep]
assert P["paper.fafix_n"] == _n2

# Tuning pilot ranges (results/tune; keys from src/score_tune.py).
_bs = sorted(int(k.split(".")[2][1:]) for k in S if k.startswith("tune.q25.b") and k.endswith(".tuned_loss"))
if _bs:
    P["paper.tune_loss_min"] = min(S[f"tune.q25.b{n}.tuned_loss"] for n in _bs)
    P["paper.tune_loss_max"] = max(S[f"tune.q25.b{n}.tuned_loss"] for n in _bs)
    P["paper.tune_stock_min"] = min(S[f"tune.q25.b{n}.stock_loss"] for n in _bs)
    P["paper.tune_stock_max"] = max(S[f"tune.q25.b{n}.stock_loss"] for n in _bs)
    P["paper.tune_bmax"] = max(_bs)
    _t = json.loads((STATS.parent / "tune" / "tuned_q25.json").read_text())
    _small = [e["ms"][m]["default"] / e["ms"][m]["tuned"] for e in _t["entries"] for m in e["ms"] if int(m) <= 32]
    _large = [e["ms"][m]["tuned"] / e["ms"][m]["cublas"] for e in _t["entries"] if e["n"] != 152064 for m in e["ms"] if int(m) >= 256]
    _near = [e["ms"][m]["tuned"] / e["ms"][m]["cublas"] for e in _t["entries"] for m in e["ms"] if int(m) <= 32]
    P["paper.tune_speedup_small_min"], P["paper.tune_speedup_small_max"] = min(_small), max(_small)
    P["paper.tune_vs_cublas_large_min"], P["paper.tune_vs_cublas_large_max"] = min(_large), max(_large)
    P["paper.tune_vs_cublas_small_min"], P["paper.tune_vs_cublas_small_max"] = min(_near), max(_near)

# Phase 1d on B200 (results/phase1d_b200, cuBLAS 13.1, bi mode): the failing set is the odd widths.
b2 = cells_dir("phase1d_b200", "bi")
h1 = cells_dir("phase1d", "bi")
assert set(b2) == set(h1)
b2_fail_n = sorted({n for (n, k), f in b2.items() if f})
assert all(n % 2 == 1 for n in b2_fail_n) and all(b2[(n, k)] for (n, k) in b2 if n % 2 == 1), "B200 failures are not exactly the odd widths"
P["paper.p1d_b200_shapes"] = len(b2)
P["paper.p1d_b200_differ_from_h100"] = sum(b2[k] != h1[k] for k in b2)
P["paper.p1d_b200_fail_min_n"], P["paper.p1d_b200_fail_max_n"] = min(b2_fail_n), max(b2_fail_n)
P["paper.p1d_b200_router_pass"] = not b2[(32, 2880)]

merge(P)
print(len(P), "paper stats")
for k, v in sorted(P.items()):
    print(" ", k, v)
