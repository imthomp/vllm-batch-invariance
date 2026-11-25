#!/usr/bin/env python3
"""Render paper/macros.tex and paper/tab_*.tex from results/stats.json. Build artifacts: never edit them by hand."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "results/stats.json").read_text())
OUT = ROOT / "paper"


def pct(v, d=1):
    s = f"{abs(v) * 100:.{d}f}\\%"
    return f"$-${s}" if v < 0 and round(abs(v) * 100, d) > 0 else s


def num(v):
    return f"{int(v):,}".replace(",", "{,}")


def lst(v):
    v = [str(int(x)) for x in v]
    return v[0] if len(v) == 1 else ", ".join(v[:-1]) + " and " + v[-1]


def times(v):
    return f"{v:.2f}"


def lenf(v):
    return num(len(v))


def frac(v):
    return f"{round(v * 100)}\\%"


# name: (key, formatter)
M = {
    "PoneBuildsHundred": ("paper.p1_builds_h100", num),
    "PoneBuildsBlackwell": ("paper.p1_builds_b200", num),
    "PoneRowsPerMode": ("paper.p1_rows_per_mode", num),
    "PoneBadTotal": ("paper.p1_bi_bad_total", num),
    "PoneComparisons": ("paper.p1_bi_comparisons", num),
    "PoneCtlMinH": ("paper.p1_ctl_min_h100", num),
    "PoneCtlMaxH": ("paper.p1_ctl_max_h100", num),
    "PoneCtlB": ("paper.p1_ctl_max_b200", num),
    "PoneCublasVersions": ("paper.p1_cublas_versions", num),
    "PtwoConfigs": ("paper.p2_configs", num),
    "PtwoPassConfigs": ("paper.p2_pass_configs", num),
    "PtwoCtlDivMin": ("paper.p2_ctl_div_min", frac),
    "PtwoCtlDivMax": ("paper.p2_ctl_div_max", frac),
    "PtwoCostCells": ("paper.p2_cost_cells", num),
    "PtwoCostPass": ("paper.p2_cost_pass", num),
    "PtwoPrefillPlainMax": ("paper.p2_prefill_plain_max", pct),
    "PtwoPrefillStockMin": ("paper.p2_prefill_stock_min", pct),
    "PtwoPrefillStockMax": ("paper.p2_prefill_stock_max", pct),
    "QsevenStock": ("phase2.decode.q25.32.on_loss_median", pct),
    "QsevenPlain": ("phase2.decode.q25.32.plain_loss_median", pct),
    "QsevenPlainMax": ("phase2.decode.q25.32.plain_loss_max", pct),
    "GemmaStock": ("phase2.decode.gemma2.32.on_loss_median", pct),
    "GemmaPlain": ("phase2.decode.gemma2.32.plain_loss_median", pct),
    "TwoPStock": ("phase2.decode.q25tp2.32.on_loss_median", pct),
    "TwoPPlain": ("phase2.decode.q25tp2.32.plain_loss_median", pct),
    "BigStock": ("phase2.decode.q32tp2.32.on_loss_median", pct),
    "BigPlain": ("phase2.decode.q32tp2.32.plain_loss_median", pct),
    "TwoPComm": ("phase2.decode.q25tp2.32.comm_loss_median", pct),
    "DiagReq": ("phase2.cross.g2tri_on.requests", num),
    "DiagTritonOn": ("phase2.cross.g2tri_on.mismatch_requests", num),
    "DiagTritonOff": ("phase2.cross.g2tri_off.mismatch_requests", num),
    "DiagEagerOn": ("phase2.cross.g2eager_on.mismatch_requests", num),
    "DiagEagerOff": ("phase2.cross.g2eager_off.mismatch_requests", num),
    "DiagNoscOn": ("phase2.cross.g2nosc_on.mismatch_requests", num),
    "DiagNoscOff": ("phase2.cross.g2nosc_off.mismatch_requests", num),
    "DiagSevenOn": ("phase2.cross.g7_on.mismatch_requests", num),
    "DiagSevenOff": ("phase2.cross.g7_off.mismatch_requests", num),
    "DiagNoswOn": ("phase2.cross.g2nosw_on.mismatch_requests", num),
    "DiagNoswOff": ("phase2.cross.g2nosw_off.mismatch_requests", num),
    "DiagNocgOn": ("phase2.cross.g2nocg_on.mismatch_requests", num),
    "DiagNocgOff": ("phase2.cross.g2nocg_off.mismatch_requests", num),
    "ShapeCount": ("paper.p1c_shapes", num),
    "ShapeMaxFailN": ("paper.p1c_max_fail_n", num),
    "ShapeMinPassN": ("paper.p1c_min_pass_n", num),
    "RouterBfFirstBad": ("paper.p1c_router_bf16_first_bad_m", num),
    "RouterFpFirstBad": ("paper.p1c_router_fp16_first_bad_m", num),
    "MidLo": ("paper.p1d_mid_lo", num),
    "MidHi": ("paper.p1d_mid_hi", num),
    "MidMult": ("paper.p1d_mid_pass_multiple", num),
    "NarrowNs": ("paper.p1d_narrow_ns", lst),
    "NarrowFailK": ("paper.p1d_narrow_fail_k", lst),
    "NarrowPassK": ("paper.p1d_narrow_pass_k", lst),
    "NoenvFailK": ("paper.p1d_noenv_fail_k", lst),
    "NoenvMinFailN": ("paper.p1d_noenv_min_fail_n", num),
    "NoenvMaxFailN": ("paper.p1d_noenv_max_fail_n", str),
    "NoenvMinPassN": ("paper.p1d_noenv_min_pass_n", str),
    "MistralStock": ("phase2.decode.mistral.32.on_loss_median", pct),
    "MistralPlain": ("phase2.decode.mistral.32.plain_loss_median", pct),
    "GemmaOnDiv": ("phase2.cross.gemma2_on.mismatch_requests", num),
    "GemmaPlainDiv": ("phase2.cross.gemma2v2_plain.mismatch_requests", num),
    "GemmaOffDiv": ("phase2.cross.gemma2_off.mismatch_requests", num),
    "GemmaReq": ("phase2.cross.gemma2_off.requests", num),
    "OffDivQsevenMain": ("cross.main_qwen25_off.mismatch_requests", num),
    "OffReqQsevenMain": ("cross.main_qwen25_off.requests", num),
    "StockMainQseven": ("main.qwen25.b32.stock_loss", pct),
    "PlainMainQseven": ("main.qwen25.b32.plain_loss", pct),
    "TorchDefaultBad": ("bitwise.torch_default_flags.bad", num),
    "TorchDefaultRows": ("bitwise.torch_default_flags.rows", num),
    "LlamaStock": ("main.llama31.b32.stock_loss", pct),
    "QthreeStock": ("main.qwen3.b32.stock_loss", pct),
    "HtwoShapes": ("paper.p1d_h200_bi_shapes", num),
    "HtwoAgree": ("paper.p1d_h200_bi_agree", num),
    "HtwoAgreeNoenv": ("paper.p1d_h200_bi_noenv_agree", num),
    "GptossTests": ("paper.gptoss_on_tests", num),
    "GptossOnPass": ("paper.gptoss_on_pass", num),
    "GptossVtwoPass": ("paper.gptoss_v2_pass", num),
    "GptossBlanketPass": ("paper.gptoss_blanket_pass", num),
    "FafixBase": ("paper.fafix_base_div", num),
    "FafixFix": ("paper.fafix_fix_div", num),
    "FafixN": ("paper.fafix_n", num),
    "TuneShapes": ("tune.q25.shapes", num),
    "TuneRows": ("tune.q25.check_rows", num),
    "TuneBad": ("tune.q25.check_bad", num),
    "TuneCrossReq": ("tune.q25.cross_requests", num),
    "TuneCrossMis": ("tune.q25.cross_mismatch_requests", num),
    "TuneStockB": ("generic2.q25gen2.b32.on_loss", pct),
    "TunedB": ("generic2.q25gen2.b32.tuned_loss", pct),
    "TuneLossMin": ("paper.tune_loss_min", pct),
    "TuneLossMax": ("paper.tune_loss_max", pct),
    "TuneStockMin": ("paper.tune_stock_min", pct),
    "TuneStockMax": ("paper.tune_stock_max", pct),
    "TuneBmax": ("paper.tune_bmax", num),
    "TuneSpeedMin": ("paper.tune_speedup_small_min", times),
    "TuneSpeedMax": ("paper.tune_speedup_small_max", times),
    "TuneLargeMin": ("paper.tune_vs_cublas_large_min", times),
    "TuneLargeMax": ("paper.tune_vs_cublas_large_max", times),
    "TuneSmallMin": ("paper.tune_vs_cublas_small_min", times),
    "TuneSmallMax": ("paper.tune_vs_cublas_small_max", times),
    "MtSystems": ("paper.mt_systems", num),
    "MtRuns": ("paper.mt_runs", num),
    "MtChangeMin": ("paper.mt_change_min", frac),
    "MtChangeMax": ("paper.mt_change_max", frac),
    "MtComparisons": ("paper.mt_comparisons", num),
    "MtFlips": ("paper.mt_flips", num),
    "MtCometRange": ("paper.mt_comet_range_max", times),
    "PopShapes": ("paper.pop_shapes", num),
    "PopBad": ("paper.pop_bad", num),
    "PopRows": ("paper.pop_rows", num),
    "PopSpeedMin": ("paper.pop_speed_min", times),
    "PopSpeedMax": ("paper.pop_speed_max", times),
    "OssGain": ("serve_gptoss.decode.gain_over_stock", pct),
    "OssWins": ("serve_gptoss.decode.tuned_wins", num),
    "OssRounds": ("serve_gptoss.decode.rounds", num),
    "OssBigGain": ("serve_gptoss120b.decode.gain_over_stock", pct),
    "OssBigWins": ("serve_gptoss120b.decode.tuned_wins", num),
    "OssStockLoss": ("serve_gptoss.decode.stock_loss_vs_off", pct),
    "OssTunedLoss": ("serve_gptoss.decode.tuned_loss_vs_off", pct),
    "BtwoShapes": ("paper.p1d_b200_shapes", num),
    "BtwoDiffer": ("paper.p1d_b200_differ_from_h100", num),
    "BtwoFailMin": ("paper.p1d_b200_fail_min_n", num),
    "BtwoFailMax": ("paper.p1d_b200_fail_max_n", num),
    "GenericTrainShapes": ("paper.generic_train_shapes", num),
    "GenericTestShapes": ("paper.generic_test_shapes", num),
    "GenericCheckRows": ("paper.generic_check_rows", num),
    "GenericCheckBad": ("paper.generic_check_bad", num),
    "GenericCrossRequests": ("paper.generic_cross_requests", num),
    "GenericCrossMismatch": ("paper.generic_cross_mismatch", num),
    "GenericRecovered": ("paper.generic_recovered_b32", frac),
    "GenericStockB": ("paper.generic_stock_b32", pct),
    "GenericOnbreakB": ("paper.generic_onbreak_b32", pct),
    "GenDefaultLossBatchThirtyTwo": ("paper.generic_b32", pct),
    "GenTunedLossBatchThirtyTwo": ("paper.generic_tuned_b32", pct),
    "GenDefaultLossBatchOneTwentyEight": ("paper.generic_b128", pct),
    "GenTunedLossBatchOneTwentyEight": ("paper.generic_tuned_b128", pct),
    "GenericGemmaStockB": ("paper.generic_gemma_stock_b32", pct),
    "GenericGemmaB": ("paper.generic_gemma_b32", pct),
    "GenericGemmaOnbreakB": ("paper.generic_gemma_onbreak_b32", pct),
    "GptossBreakDecGain": ("paper.gptoss_break_decode_gain", pct),
    "GptossBreakDecWins": ("paper.gptoss_break_decode_wins", num),
    "GptossBreakRounds": ("paper.gptoss_break_rounds", num),
    "GptossBreakPreGain": ("paper.gptoss_break_prefill_gain", pct),
    "GptossBreakPreWins": ("paper.gptoss_break_prefill_wins", num),
    "GptossNoBreakDecGain": ("paper.gptoss_no_break_decode_gain", pct),
    "GptossNoBreakPreGain": ("paper.gptoss_no_break_prefill_gain", pct),
    "CoreQsevenDec": ("paper.core_q25_decode32", pct),
    "CoreQsevenDecOne": ("paper.core_q25_decode1", pct),
    "CoreUntunedDecMin": ("paper.core_untuned_decode32_min", pct),
    "CoreUntunedDecMax": ("paper.core_untuned_decode32_max", pct),
    "CoreGemmaPreLoss": ("paper.core_gemma_prefill32_loss", pct),
    "CoreGemmaPreWins": ("paper.core_gemma_prefill32_wins", num),
    "CoreMistralDec": ("paper.core_mistral_decode32", pct),
    "CoreRounds": ("paper.core_rounds", num),
    "DiagGemmaRounds": ("paper.gemmadiag_rounds", num),
    "DiagGemmaBasebrkPre": ("paper.gemmadiag_basebrk_prefill32", pct),
    "DiagGemmaBasebrkPreWins": ("paper.gemmadiag_basebrk_prefill32_wins", num),
    "DiagGemmaBasebrkDec": ("paper.gemmadiag_basebrk_decode32", pct),
    "DiagGemmaPrPre": ("paper.gemmadiag_pr_prefill32", pct),
    "DiagGemmaPrnobrkPre": ("paper.gemmadiag_prnobrk_prefill32", pct),
    "DiagGemmaPrnobrkPreWins": ("paper.gemmadiag_prnobrk_prefill32_wins", num),
    "DiagGemmaPrnobrkDec": ("paper.gemmadiag_prnobrk_decode32", pct),
    "DiagGemmaPrDec": ("paper.gemmadiag_pr_decode32", pct),
    "OssGateRounds": ("paper.ossgate_rounds", num),
    "OssGateDecOneMin": ("paper.ossgate_decode1_min", pct),
    "OssGateDecOneMax": ("paper.ossgate_decode1_max", pct),
    "OssGateDecOneWins": ("paper.ossgate_decode1_wins_min", num),
    "OssGateDecMin": ("paper.ossgate_decode32_min", pct),
    "OssGateDecMax": ("paper.ossgate_decode32_max", pct),
    "OssGateDecWins": ("paper.ossgate_decode32_wins_min", num),
    "OssGatePreAbsMax": ("paper.ossgate_prefill32_absmax", pct),
}

# Results still being produced: render a visible stub until the key exists (filled by src/score_*.py).
OPTIONAL = {
    "MtvSysN": ("mtvar.sys_comparisons_n", num),
    "MtvHighBetween": ("paper.mtv_high_between", pct),
    "MtvLowBetween": ("paper.mtv_low_between", pct),
    "MtvSysCond": ("mtvar.sys_comparisons_cond", num),
    "MtvSysRepeat": ("mtvar.sys_comparisons_repeat", num),
    "MtvRankCells": ("mtvar.sys_rankings_changed", num),
    "MtvJudgeN": ("mtvar.judge_comparisons_n", num),
    "MtvJudgeCond": ("mtvar.judge_comparisons_cond", num),
    "MtvJudgeSign": ("mtvar.judge_comparisons_cond_sign", num),
    "MtvJudgeSig": ("paper.mtv_judge_sig_changes", num),
    "MtvJudgeHolmCond": ("paper.mtv_judge_holm_cond", num),
    "MtvJudgeHolmSig": ("paper.mtv_judge_holm_sig", num),
    "MtvJudgeRepeat": ("mtvar.judge_comparisons_repeat", num),
    "MtvJudgeBetween": ("mtvar.judge.change_between_mean", pct),
    "MtvJudgeWithin": ("mtvar.judge.change_within_mean", pct),
    "MtvJudgeScoreChange": ("mtvar.judge.score_change_between_mean", pct),
    "MtvBetweenMin": ("paper.mtv_between_min", pct),
    "MtvBetweenMax": ("paper.mtv_between_max", pct),
    "MtvWithinMax": ("paper.mtv_within_max", pct),
    "MtvTimeMin": ("paper.mtv_time_min", pct),
    "MtvTimeMax": ("paper.mtv_time_max", pct),
    "MtvBiDiffer": ("paper.mtv_bi_differ_total", num),
    "MtvSideMxShare": ("paper.mtv_side_metricx_share", pct),
    "MtvSideMxMax": ("mtvar.side.metricx.bs1_vs_bs16.max_abs", times),
    "MtvSideCometShare": ("paper.mtv_side_comet22_share", pct),
    "MtvSideCometMax": ("mtvar.side.comet22.bs1_vs_bs16.max_abs", times),
    "EvComparisons": ("evalvar.comparisons", num),
    "EvChangedAny": ("evalvar.comparisons_changed_any", num),
    "EvChangedRepeat": ("evalvar.comparisons_changed_repeat", num),
    "EvChangedCond": ("evalvar.comparisons_changed_condition_only", num),
    "EvRankGsm": ("evalvar.gsm8k.distinct_rankings", num),
    "EvRankMath": ("evalvar.math500.distinct_rankings", num),
    "EvFlipBetweenMin": ("paper.ev_flip_between_min", pct),
    "EvFlipBetweenMax": ("paper.ev_flip_between_max", pct),
    "EvFlipWithinMax": ("paper.ev_flip_within_max", pct),
    "EvSpreadMax": ("paper.ev_spread_max", pct),
    "EvRepeatSpreadMax": ("paper.ev_repeat_spread_max", pct),
    "EvBiDiffer": ("paper.ev_bi_differ_total", num),
    "EvTimeCOneMin": ("paper.ev_time_C1_min", pct),
    "EvTimeCOneMax": ("paper.ev_time_C1_max", pct),
    "EvTimeCTwoMin": ("paper.ev_time_C2_min", pct),
    "EvTimeCTwoMax": ("paper.ev_time_C2_max", pct),
    "OneEVerdict": ("p1e.verdict", str),
    "OneEShapes": ("p1e.C.shapes", num),
    "OneEBadC": ("p1e.C.bad", num),
    "OneECompC": ("p1e.C.comparisons", num),
    "OneEErrC": ("p1e.C.error_shapes", lenf),
    "OneEBadB": ("p1e.B.bad", num),
    "OneEBadShapesB": ("p1e.B.bad_shapes", lenf),
    "OneEBadD": ("p1e.D.bad", num),
    "OneECompD": ("p1e.D.comparisons", num),
    "OneECompE": ("p1e.E.comparisons", num),
    "OneEBadShapesC": ("p1e.C.bad_shapes", lenf),
    "OneEBadE": ("p1e.E.bad", num),
    "OneEErrA": ("p1e.A.error_shapes", lenf),
    "FAutoBad": ("p1f.auto.bad", num),
    "FAutoComp": ("p1f.auto.comparisons", num),
    "FAlignTwoBad": ("p1f.align2.bad", num),
    "FAlignSixteenErr": ("p1f.align16.errors", num),
    "FLtBad": ("p1f.auto_lt.bad", num),
    "FLtComp": ("p1f.auto_lt.comparisons", num),
    "ServeUnit": ("serve.unit_passed", num),
    "ServeRounds": ("serve.decode.rounds", num),
    "ServeWins": ("serve.decode.tuned_wins", num),
    "ServeDecGain": ("serve.decode.gain_over_stock", pct),
    "ServeDecStockLoss": ("serve.decode.stock_loss_vs_off", pct),
    "ServeDecTunedLoss": ("serve.decode.tuned_loss_vs_off", pct),
    "ServePreGain": ("serve.prefill.gain_over_stock", pct),
    "ServePreTunedLoss": ("serve.prefill.tuned_loss_vs_off", pct),
}

lines = ["% Generated by src/make_macros.py from results/stats.json. Do not edit."]
for name, (key, f) in M.items():
    if key not in S:
        sys.exit(f"missing stat {key} for \\{name}")
    lines.append(f"\\newcommand{{\\{name}}}{{{f(S[key])}}}")
pending = []
for name, (key, f) in OPTIONAL.items():
    if key in S:
        lines.append(f"\\newcommand{{\\{name}}}{{{f(S[key])}}}")
    else:
        pending.append(name)
        lines.append(f"\\newcommand{{\\{name}}}{{\\pending{{{key.replace('_', '-')}}}}}")
if pending:
    print("pending stubs:", " ".join(pending))
(OUT / "macros.tex").write_text("\n".join(lines) + "\n")

DET = [("q25tp2", "Qwen2.5-7B, TP=2", "q25tp2"), ("q25tp4", "Qwen2.5-7B, TP=4 (TP=2 control)", "q25tp2"),
       ("q25long", "Qwen2.5-7B, long context, chunked prefill", "q25long"),
       ("q25samp", "Qwen2.5-7B, seeded sampling", "q25samp"), ("q32tp2", "Qwen2.5-32B, TP=2", "q32tp2"),
       ("mistralv2", "Mistral-7B", "mistral"), ("gemma2v2", "Gemma-2-9B", "gemma2")]
t = ["% Generated by src/make_macros.py. Do not edit.", "\\begin{tabular}{lrrr}", "\\toprule",
     "Configuration & Plain cuBLAS & Mode off & Markers \\\\", "\\midrule"]
for tag, label, ctl in DET:
    k, c = f"phase2.cross.{tag}_plain", f"phase2.cross.{ctl}_off"
    assert (ROOT / "results" / "phase2" / f"cross_{tag}_plain.jsonl.gz").is_file(), f"missing source rows for {label}"
    assert (ROOT / "results" / "phase2" / f"cross_{ctl}_off.jsonl.gz").is_file(), f"missing control rows for {label}"
    t.append(f"{label} & {num(S[k + '.mismatch_requests'])} / {num(S[k + '.requests'])} & "
             f"{num(S[c + '.mismatch_requests'])} / {num(S[c + '.requests'])} & {num(S[k + '.markers'])} \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
(OUT / "tab_det.tex").write_text("\n".join(t) + "\n")

COST = [("q25", "Qwen2.5-7B"), ("gemma2", "Gemma-2-9B"), ("mistral", "Mistral-7B"),
        ("q25tp2", "Qwen2.5-7B, TP=2"), ("q32tp2", "Qwen2.5-32B, TP=2")]
t = ["% Generated by src/make_macros.py. Do not edit.", "\\begin{tabular}{lrrr}", "\\toprule",
     "Model & Stock & Plain cuBLAS & Plain, worst repeat \\\\", "\\midrule"]
for tag, label in COST:
    k = f"phase2.decode.{tag}.32"
    for arm in ("off", "on", "plain"):
        runs = sorted((ROOT / "results" / "phase2").glob(f"decode_{tag}_{arm}_r*.json"))
        count_key = ".plain_n" if arm == "plain" else ".on_n"
        assert len(runs) == S[k + count_key], f"source run count mismatch for {label}/{arm}"
        for run in runs:
            rows = json.loads(run.read_text()).get("rows", [])
            assert any(r.get("n") == 32 for r in rows), f"{run.name} has no batch-32 record"
    t.append(f"{label} & {pct(S[k + '.on_loss_median'])} & {pct(S[k + '.plain_loss_median'])} & "
             f"{pct(S[k + '.plain_loss_max'])} \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
(OUT / "tab_cost.tex").write_text("\n".join(t) + "\n")

t = ["% Generated by src/make_macros.py. Do not edit.", "\\begin{tabular}{lrrr}", "\\toprule",
     "GPU, torch build & BI flags & Flags only & Defaults \\\\", "\\midrule"]
tags = sorted({k[len("phase1."):-len(".bi.bad")] for k in S if k.startswith("phase1.") and k.endswith(".bi.bad")})
for tag in tags:
    gpu, rest = tag.split("_", 1)
    for arm in ("bi", "bi_noenv", "default"):
        src = ROOT / "results" / "phase1" / f"{tag}_{arm}.json"
        assert src.is_file() and json.loads(src.read_text()).get("rows"), f"missing source records for {tag}/{arm}"
    label = f"{gpu.upper()}, torch {rest.replace('_stress', ' + GPU load').replace('_', '+')}"
    t.append(f"{label} & {num(S[f'phase1.{tag}.bi.bad'])} & {num(S[f'phase1.{tag}.bi_noenv.bad'])} & "
             f"{num(S[f'phase1.{tag}.default.bad'])} \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
(OUT / "tab_bitwise.tex").write_text("\n".join(t) + "\n")
print("wrote macros.tex,", len(M), "macros; tab_det.tex, tab_cost.tex, tab_bitwise.tex")

# Better-default rerun: decode loss against mode off, per batch size, medians of three jobs per arm (results/generic2).
bs = sorted(int(k.split(".")[2][1:]) for k in S if k.startswith("generic2.q25gen2.b") and k.endswith(".tuned_loss"))
assert bs and len(bs) == 8
for arm in ("off", "on", "onbreak", "generic", "tuned"):
    runs = sorted((ROOT / "results" / "generic2").glob(f"decode_q25gen2_{arm}_r*.json"))
    assert len(runs) == 3, f"table arm {arm} has {len(runs)} source runs, expected the three-repeat rerun"
    for run in runs:
        rows = json.loads(run.read_text()).get("rows", [])
        assert {r.get("n") for r in rows} >= set(bs), f"{run.name} lacks a table batch-size record"
t = ["% Generated by src/make_macros.py from results/stats.json; each row is checked against results/generic2 run records.", "\\begin{tabular}{l" + "r" * len(bs) + "}", "\\toprule",
     "Batch size & " + " & ".join(num(b) for b in bs) + " \\\\", "\\midrule",
     "Stock mode (default config) & " + " & ".join(pct(S[f"generic2.q25gen2.b{b}.on_loss"]) for b in bs) + " \\\\",
     "Stock + breakable & " + " & ".join(pct(S[f"generic2.q25gen2.b{b}.onbreak_loss"]) for b in bs) + " \\\\",
     "Generic + breakable & " + " & ".join(pct(S[f"generic2.q25gen2.b{b}.generic_loss"]) for b in bs) + " \\\\",
     "Per-model tuned table & " + " & ".join(pct(S[f"generic2.q25gen2.b{b}.tuned_loss"]) for b in bs) + " \\\\",
     "\\bottomrule", "\\end{tabular}"]
(OUT / "tab_tune.tex").write_text("\n".join(t) + "\n")
print("wrote tab_tune.tex")

# Core PR #67376 vs its merge-base, end-to-end serving (results/validation/core_pr_serving; src/score_core_pr_serve.py).
# Each cell is the median of three paired per-round PR/base ratios; every row is checked against its raw bench records.
CORE = [("h100_q25", "Qwen2.5-7B, H100"), ("h200_q25", "Qwen2.5-7B, H200"), ("h100_qwen32", "Qwen2.5-32B, TP=2, H100"),
        ("h100_gemma", "Gemma-2-9B, H100"), ("h100_mistral", "Mistral-7B, H100")]
CORE_W = [("decode32", "Decode, c32"), ("prefill32", "Prefill, c32"), ("decode1", "Decode, c1"), ("longdecode32", "Long prompt, c32")]


def spct(v):
    return ("+" if v > 0 else "") + pct(v)


t = ["% Generated by src/make_macros.py from results/stats.json; each row is checked against its raw vllm bench serve records.",
     "\\begin{tabular}{l" + "r" * len(CORE_W) + "}", "\\toprule",
     "Model, GPU & " + " & ".join(lbl for _, lbl in CORE_W) + " \\\\", "\\midrule"]
for tag, label in CORE:
    d = ROOT / "results" / "validation" / "core_pr_serving" / tag
    meta = json.loads((d / "metadata.json").read_text())
    assert meta["base_sha"] == S[f"core_pr_serve.{tag}.base_sha"] and meta["pr_sha"] == S[f"core_pr_serve.{tag}.pr_sha"], tag
    cells = []
    for w, _ in CORE_W:
        k = f"core_pr_serve.{tag}.{w}"
        n = S[k + ".rounds"]
        assert n == 3, f"{tag}/{w}: {n} rounds"
        for arm in ("base", "pr", "off"):
            for r in range(1, n + 1):
                rec = json.loads((d / f"bench_{w}_{arm}_r{r}.json").read_text())
                assert rec.get("completed", 0) > 0, f"{tag}/{w}/{arm}/r{r} has no completed requests"
        cells.append(f"{spct(S[k + '.pr_vs_base_median'])} ({num(S[k + '.pr_wins'])}/{num(n)})")
    t.append(f"{label} & " + " & ".join(cells) + " \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
(OUT / "tab_corepr.tex").write_text("\n".join(t) + "\n")
print("wrote tab_corepr.tex")

# Eval-variance table (one row per model). Every row is checked against its 16 archived runs before it is written.
EV = [("qwen25_7b", "Qwen2.5-7B"), ("llama31_8b", "Llama-3.1-8B"), ("gemma2_9b", "Gemma-2-9B"),
      ("mistral_7b", "Mistral-7B"), ("olmo2_7b", "OLMo-2-7B"), ("phi4_mini", "Phi-4-mini")]
if "evalvar.comparisons" in S:
    def acc_rng(m, b):
        v = [S[f"evalvar.{m}.{b}.acc_{c}_r{r}"] for c in ("C1", "C2", "C3", "C4") for r in (1, 2)]
        return f"{min(v) * 100:.1f}--{max(v) * 100:.1f}"
    t = ["% Generated by src/make_macros.py from results/stats.json; each row is checked against its archived runs.",
         "\\begin{tabular}{lrrrrrr}", "\\toprule",
         "Model & GSM8K acc. & MATH-500 acc. & Flips between & Flips within & BI-on differing & BI-on time (C2) \\\\",
         "\\midrule"]
    for m, label in EV:
        d = ROOT / "results" / "validation" / "evalvar" / m
        runs = sorted(d.glob("bi*/C*_r*.jsonl.gz"))
        assert len(runs) == 16 and all(r.with_name(r.name[:-9] + ".meta.json").exists() for r in runs), f"{m}: {len(runs)} runs"
        fb = "/".join(pct(S[f"evalvar.{m}.{b}.flip_between_mean"]) for b in ("gsm8k", "math500"))
        fw = "/".join(pct(S[f"evalvar.{m}.{b}.flip_within_mean"]) for b in ("gsm8k", "math500"))
        t.append(f"{label} & {acc_rng(m, 'gsm8k')} & {acc_rng(m, 'math500')} & {fb} & {fw} & "
                 f"{num(S[f'evalvar.{m}.bi_on_requests_differing'])} & {spct(S[f'evalvar.{m}.C2.time_on_vs_off'])} \\\\")
    t += ["\\bottomrule", "\\end{tabular}"]
else:
    t = ["% Generated by src/make_macros.py: eval-variance results pending (src/evalvar_score.py --merge).",
         "\\pending{eval-variance table}"]
(OUT / "tab_evalvar.tex").write_text("\n".join(t) + "\n")
print("wrote tab_evalvar.tex")

# MT eval-variance table: one row per translation system plus the judge; every row checked against its 16 archived runs.
MTV = [("gemma2_9b", "Gemma-2-9B"), ("gemma3_12b", "Gemma-3-12B"), ("aya_8b", "Aya-Expanse-8B"),
       ("qwen25_7b", "Qwen2.5-7B"), ("llama31_8b", "Llama-3.1-8B")]
if "mtvar.sys_comparisons_n" in S:
    t = ["% Generated by src/make_macros.py from results/stats.json; each row is checked against its archived runs.",
         "\\begin{tabular}{lrrrr}", "\\toprule",
         "Model & Changed, between & Changed, within & BI-on differing & BI-on time (C2) \\\\", "\\midrule"]
    def rows_ok(task, m):
        d = ROOT / "results" / "validation" / "mtvar" / task / m
        runs = sorted(d.glob("bi*/C*_r*.jsonl.gz"))
        assert len(runs) == 16 and all(r.with_name(r.name[:-9] + ".meta.json").exists() for r in runs), (task, m, len(runs))
    for m, label in MTV:
        rows_ok("translate", m)
        t.append(f"{label} & {pct(S[f'paper.mtv_{m}_between'])} & {pct(S[f'paper.mtv_{m}_within'])} & "
                 f"{num(S[f'mtvar.{m}.bi_on_requests_differing'])} & {spct(S[f'mtvar.{m}.C2.time_on_vs_off'])} \\\\")
    rows_ok("judge", "qwen25_14b")
    t += ["\\midrule", f"Judge (Qwen2.5-14B) & {pct(S['mtvar.judge.change_between_mean'])} & {pct(S['mtvar.judge.change_within_mean'])} & "
          f"{num(S['mtvar.qwen25_14b.bi_on_requests_differing'])} & {spct(S['mtvar.qwen25_14b.C2.time_on_vs_off'])} \\\\",
          "\\bottomrule", "\\end{tabular}"]
else:
    t = ["% Generated by src/make_macros.py: MT eval-variance results pending.", "\\pending{MT eval-variance table}"]
(OUT / "tab_mtvar.tex").write_text("\n".join(t) + "\n")
print("wrote tab_mtvar.tex")
