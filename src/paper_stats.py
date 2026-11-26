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

# MT pilot (results/pilot_mt/mt_summary.json, from src/mt_analyze.py; rule in results/pilot_mt/PREREG_mt.md).
_mt = json.loads((STATS.parent / "pilot_mt" / "mt_summary.json").read_text())
_ch = [v["mean"] for v in _mt["change"].values()]
P["paper.mt_systems"] = len(_mt["systems"])
P["paper.mt_runs"] = len(_mt["runs"])
P["paper.mt_change_min"], P["paper.mt_change_max"] = min(_ch), max(_ch)
P["paper.mt_comparisons"] = len(_mt["comparisons"])
P["paper.mt_flips"] = sum(c["sign_flip"] or c["sig_flip"] for c in _mt["comparisons"])
P["paper.mt_comet_range_max"] = max(v["range"] for k, v in _mt["range"].items() if k.startswith("comet|"))
P["paper.mt_go"] = _mt["go"]

# Popular-model tuning (results/tune: GLM-5.3, DeepSeek-V4.1, gpt-oss-20b/120b shapes; PR #65931).
_pe, _pb, _pr = [], 0, 0
for _f in ("tuned_popular.json", "tuned_gptoss20b.json", "tuned_gptoss120b_router.json"):
    _t = json.loads((STATS.parent / "tune" / _f).read_text())
    _pe += _t["entries"]; _pb += sum(v["bad"] for v in _t["check"].values()); _pr += sum(v["rows"] for v in _t["check"].values())
_sp = [e["ms"][m]["default"] / e["ms"][m]["tuned"] for e in _pe for m in e["ms"]]
P["paper.pop_shapes"], P["paper.pop_bad"], P["paper.pop_rows"] = len(_pe), _pb, _pr
P["paper.pop_speed_min"], P["paper.pop_speed_max"] = min(_sp), max(_sp)
assert _pb == 0 and min(_sp) > 1

# Better default rerun (results/generic2): all paper values come from the scored, three-repeat JSON records.
_g = json.loads((STATS.parent / "generic2" / "generic.json").read_text())
assert len(_g["test"]) == S["generic2.test_shapes"]
assert sum(v["rows"] for v in _g["check"].values()) == S["generic2.check_rows"]
assert sum(v["bad"] for v in _g["check"].values()) == S["generic2.check_bad"]
P["paper.generic_train_shapes"] = len(_g["train"])
P["paper.generic_test_shapes"] = len(_g["test"])
P["paper.generic_check_rows"] = sum(v["rows"] for v in _g["check"].values())
P["paper.generic_check_bad"] = sum(v["bad"] for v in _g["check"].values())
P["paper.generic_cross_requests"] = S["generic2.cross_requests"]
P["paper.generic_cross_mismatch"] = S["generic2.cross_mismatch"]
_bs = sorted(int(k.split(".")[2][1:]) for k in S if k.startswith("generic2.q25gen2.b") and k.endswith(".generic_loss"))
assert _bs == [1, 2, 4, 8, 16, 32, 64, 128]
for _b in _bs:
    for _arm in ("on", "onbreak", "generic", "tuned"):
        assert S[f"generic2.q25gen2.b{_b}.{_arm}_loss"] is not None
P["paper.generic_stock_min"] = min(S[f"generic2.q25gen2.b{b}.on_loss"] for b in _bs)
P["paper.generic_stock_max"] = max(S[f"generic2.q25gen2.b{b}.on_loss"] for b in _bs)
P["paper.generic_tuned_min"] = min(S[f"generic2.q25gen2.b{b}.tuned_loss"] for b in _bs)
P["paper.generic_tuned_max"] = max(S[f"generic2.q25gen2.b{b}.tuned_loss"] for b in _bs)
P["paper.generic_recovered_b32"] = S["generic2.q25.b32.recovered"]
P["paper.generic_b32"] = S["generic2.q25gen2.b32.generic_loss"]
P["paper.generic_tuned_b32"] = S["generic2.q25gen2.b32.tuned_loss"]
P["paper.generic_stock_b32"] = S["generic2.q25gen2.b32.on_loss"]
P["paper.generic_onbreak_b32"] = S["generic2.q25gen2.b32.onbreak_loss"]
P["paper.generic_b128"] = S["generic2.q25gen2.b128.generic_loss"]
P["paper.generic_tuned_b128"] = S["generic2.q25gen2.b128.tuned_loss"]
P["paper.generic_gemma_stock_b32"] = S["generic2.g2gen2.b32.on_loss"]
P["paper.generic_gemma_b32"] = S["generic2.g2gen2.b32.generic_loss"]
P["paper.generic_gemma_onbreak_b32"] = S["generic2.g2gen2.b32.onbreak_loss"]
# The new three-repeat series supersedes the one-job tuning-pilot ranges in the manuscript.
P["paper.tune_loss_min"] = P["paper.generic_tuned_min"]
P["paper.tune_loss_max"] = P["paper.generic_tuned_max"]
P["paper.tune_stock_min"] = P["paper.generic_stock_min"]
P["paper.tune_stock_max"] = P["paper.generic_stock_max"]

# Branch-build gpt-oss-20b recheck with breakable CUDA graphs, scored from results/branch65931/oss20_break.
_break = "branch_oss20_break"
assert S[f"{_break}.decode.stock_n"] == S[f"{_break}.decode.tuned_n"] == 2
assert S[f"{_break}.prefill.stock_n"] == S[f"{_break}.prefill.tuned_n"] == 2
P["paper.gptoss_break_decode_gain"] = S[f"{_break}.decode.gain_over_stock"]
P["paper.gptoss_break_decode_wins"] = S[f"{_break}.decode.tuned_wins"]
P["paper.gptoss_break_rounds"] = S[f"{_break}.decode.rounds"]
P["paper.gptoss_break_prefill_gain"] = S[f"{_break}.prefill.gain_over_stock"]
P["paper.gptoss_break_prefill_wins"] = S[f"{_break}.prefill.tuned_wins"]
P["paper.gptoss_no_break_decode_gain"] = S["branch_oss20_v2.decode.gain_over_stock"]
P["paper.gptoss_no_break_prefill_gain"] = S["branch_oss20_v2.prefill.gain_over_stock"]

# Core PR #67376 vs its merge-base (results/validation/core_pr_serving; src/score_core_pr_serve.py), medians of three paired rounds.
_c = "core_pr_serve"
_untuned = ("h100_q25", "h200_q25", "h100_qwen32", "h100_gemma")  # models whose shapes have no tuned entries
for _t in _untuned + ("h100_mistral",):
    for _w in ("decode32", "prefill32", "decode1", "longdecode32"):
        assert S[f"{_c}.{_t}.{_w}.rounds"] == 3, (_t, _w)
P["paper.core_q25_decode32"] = S[f"{_c}.h100_q25.decode32.pr_vs_base_median"]
P["paper.core_q25_decode1"] = S[f"{_c}.h100_q25.decode1.pr_vs_base_median"]
P["paper.core_untuned_decode32_min"] = min(S[f"{_c}.{t}.decode32.pr_vs_base_median"] for t in _untuned)
P["paper.core_untuned_decode32_max"] = max(S[f"{_c}.{t}.decode32.pr_vs_base_median"] for t in _untuned)
P["paper.core_gemma_prefill32_loss"] = -S[f"{_c}.h100_gemma.prefill32.pr_vs_base_median"]
P["paper.core_gemma_prefill32_wins"] = S[f"{_c}.h100_gemma.prefill32.pr_wins"]
P["paper.core_mistral_decode32"] = S[f"{_c}.h100_mistral.decode32.pr_vs_base_median"]
P["paper.core_rounds"] = S[f"{_c}.h100_q25.decode32.rounds"]

# Gemma-2-9B prefill diagnosis (PROTOCOL.md "Diagnostic: Gemma-2-9B prefill regression"; src/score_followup.py), a separate
# H100 run from the core-PR table: arms base, basebrk (base + forced breakable graphs), pr, prnobrk (pr with breakable off).
_g = "followup.h100_gemma_prefill_diag"
P["paper.gemmadiag_rounds"] = S[f"{_g}.rounds"]
for _arm in ("basebrk", "pr", "prnobrk"):
    for _w in ("prefill32", "decode32"):
        P[f"paper.gemmadiag_{_arm}_{_w}"] = S[f"{_g}.{_w}.{_arm}_vs_base_median"]
        P[f"paper.gemmadiag_{_arm}_{_w}_wins"] = S[f"{_g}.{_w}.{_arm}_vs_base_wins"]

# gpt-oss-20b follow-up to #65931 (PROTOCOL.md "gpt-oss follow-up to #65931"): gate (MXFP4 auto-enables breakable graphs)
# vs cfg (#65931 configs only), H100 and H200, three rounds each.
_o = [f"followup.{t}_oss20_v2" for t in ("h100", "h200")]
assert all(S[f"{t}.rounds"] == 3 and S[f"{t}.valid"] for t in _o)
P["paper.ossgate_rounds"] = 3
for _w in ("decode1", "decode32", "longdecode32", "prefill32"):
    _v = [S[f"{t}.{_w}.gate_vs_cfg_median"] for t in _o]
    P[f"paper.ossgate_{_w}_min"], P[f"paper.ossgate_{_w}_max"] = min(_v), max(_v)
    P[f"paper.ossgate_{_w}_wins_min"] = min(S[f"{t}.{_w}.gate_vs_cfg_wins"] for t in _o)
P["paper.ossgate_prefill32_absmax"] = max(abs(S[f"{t}.prefill32.gate_vs_cfg_median"]) for t in _o)

# Eval-variance experiment (PROTOCOL.md "Do serving conditions change NLP evaluation conclusions?"; src/evalvar_score.py).
# Derived only once the scorer has merged its keys; macros render as visible PENDING stubs until then.
_EVM = ("qwen25_7b", "llama31_8b", "gemma2_9b", "mistral_7b", "olmo2_7b", "phi4_mini")
if "evalvar.comparisons" in S:
    assert S["evalvar.comparisons"] == 30, S["evalvar.comparisons"]
    _mb = [(m, b) for m in _EVM for b in ("gsm8k", "math500")]
    _fb = [S[f"evalvar.{m}.{b}.flip_between_mean"] for m, b in _mb]
    P["paper.ev_flip_between_min"], P["paper.ev_flip_between_max"] = min(_fb), max(_fb)
    P["paper.ev_flip_within_max"] = max(S[f"evalvar.{m}.{b}.flip_within_max"] for m, b in _mb)
    P["paper.ev_spread_max"] = max(S[f"evalvar.{m}.{b}.acc_spread_all"] for m, b in _mb)
    P["paper.ev_repeat_spread_max"] = max(S[f"evalvar.{m}.{b}.acc_spread_repeat_max"] for m, b in _mb)
    P["paper.ev_bi_differ_total"] = sum(S[f"evalvar.{m}.bi_on_requests_differing"] for m in _EVM)
    for _c in ("C1", "C2"):
        _t = [S[f"evalvar.{m}.{_c}.time_on_vs_off"] for m in _EVM]
        P[f"paper.ev_time_{_c}_min"], P[f"paper.ev_time_{_c}_max"] = min(_t), max(_t)

# MT eval-variance study (PROTOCOL.md "Do serving conditions change MT evaluation conclusions?"; src/mtvar_score.py).
_MS = ("gemma2_9b", "gemma3_12b", "aya_8b", "qwen25_7b", "llama31_8b")
_ML = ("de_DE", "zh_CN", "ja_JP", "is_IS", "sw_KE", "gu_IN", "zu_ZA")
if "mtvar.sys_comparisons_n" in S:
    assert S["mtvar.sys_comparisons_n"] == 190 and S["mtvar.judge_comparisons_n"] == 40
    for _s in _MS:
        P[f"paper.mtv_{_s}_between"] = sum(S[f"mtvar.{_s}.{lp}.change_between_mean"] for lp in _ML) / len(_ML)
        P[f"paper.mtv_{_s}_within"] = sum(S[f"mtvar.{_s}.{lp}.change_within_mean"] for lp in _ML) / len(_ML)
    P["paper.mtv_between_min"] = min(P[f"paper.mtv_{_s}_between"] for _s in _MS)
    P["paper.mtv_between_max"] = max(P[f"paper.mtv_{_s}_between"] for _s in _MS)
    P["paper.mtv_within_max"] = max(P[f"paper.mtv_{_s}_within"] for _s in _MS)
    P["paper.mtv_judge_sig_changes"] = S["mtvar.judge_comparisons_cond"] - S["mtvar.judge_comparisons_cond_sign"]
    P["paper.mtv_judge_holm_cond"] = S["mtvar.multiplicity.holm_global.condition_sensitive"]
    P["paper.mtv_judge_holm_sig"] = S["mtvar.multiplicity.holm_global.significance_changed"]
    assert P["paper.mtv_judge_holm_cond"] == S["mtvar.multiplicity.holm_global.sign_changed"]
    assert P["paper.mtv_judge_holm_sig"] == 0
    _t = [S[f"mtvar.{m}.{c}.time_on_vs_off"] for m in _MS + ("qwen25_14b",) for c in ("C1", "C2", "C3", "C4")]
    P["paper.mtv_time_min"], P["paper.mtv_time_max"] = min(_t), max(_t)
    P["paper.mtv_bi_differ_total"] = sum(S[f"mtvar.{m}.bi_on_requests_differing"] for m in _MS + ("qwen25_14b",))
    # Descriptive grouping by resource level (directions were chosen as 3 high- and 4 lower-resource; not a pre-registered test)
    P["paper.mtv_high_between"] = sum(S[f"mtvar.{s}.{lp}.change_between_mean"] for s in _MS for lp in _ML[:3]) / (len(_MS) * 3)
    P["paper.mtv_low_between"] = sum(S[f"mtvar.{s}.{lp}.change_between_mean"] for s in _MS for lp in _ML[3:]) / (len(_MS) * 4)
    for _m, _n in (("metricx", "n_metricx"), ("comet22", "n_comet")):
        P[f"paper.mtv_side_{_m}_share"] = S[f"mtvar.side.{_m}.bs1_vs_bs16.differ"] / S[f"mtvar.side.{_n}"]

# Preregistered robustness extension summaries. Counts remain in the canonical stats file; only derived rates
# belong here. A5 is available as soon as the three same-GPU J5 metric score files have been analyzed.
if "mtvar.a5.comparisons_n" in S:
    assert S["mtvar.a5.comparisons_n"] == 190
    P["paper.mtv_a5_condition_sensitive"] = S["mtvar.a5.comparisons_condition_sensitive"]
    P["paper.mtv_a5_repeat_unstable"] = S["mtvar.a5.comparisons_repeat_unstable"]
    P["paper.mtv_a5_sign_changed"] = S["mtvar.a5.comparisons_condition_sign_changed"]

merge(P)
print(len(P), "paper stats")
for k, v in sorted(P.items()):
    print(" ", k, v)
