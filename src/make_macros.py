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
    "TuneStockB": ("tune.q25.b32.stock_loss", pct),
    "TunedB": ("tune.q25.b32.tuned_loss", pct),
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
}

# Results still being produced: render a visible stub until the key exists (filled by src/score_*.py).
OPTIONAL = {
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
    t.append(f"{label} & {pct(S[k + '.on_loss_median'])} & {pct(S[k + '.plain_loss_median'])} & "
             f"{pct(S[k + '.plain_loss_max'])} \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
(OUT / "tab_cost.tex").write_text("\n".join(t) + "\n")

t = ["% Generated by src/make_macros.py. Do not edit.", "\\begin{tabular}{lrrr}", "\\toprule",
     "GPU, torch build & BI flags & Flags only & Defaults \\\\", "\\midrule"]
tags = sorted({k[len("phase1."):-len(".bi.bad")] for k in S if k.startswith("phase1.") and k.endswith(".bi.bad")})
for tag in tags:
    gpu, rest = tag.split("_", 1)
    label = f"{gpu.upper()}, torch {rest.replace('_stress', ' + GPU load').replace('_', '+')}"
    t.append(f"{label} & {num(S[f'phase1.{tag}.bi.bad'])} & {num(S[f'phase1.{tag}.bi_noenv.bad'])} & "
             f"{num(S[f'phase1.{tag}.default.bad'])} \\\\")
t += ["\\bottomrule", "\\end{tabular}"]
(OUT / "tab_bitwise.tex").write_text("\n".join(t) + "\n")
print("wrote macros.tex,", len(M), "macros; tab_det.tex, tab_cost.tex, tab_bitwise.tex")

# Tuning pilot: decode loss against mode off, per batch size, one job per arm (results/tune).
bs = sorted(int(k.split(".")[2][1:]) for k in S if k.startswith("tune.q25.b") and k.endswith(".tuned_loss"))
t = ["% Generated by src/make_macros.py. Do not edit.", "\\begin{tabular}{l" + "r" * len(bs) + "}", "\\toprule",
     "Batch size & " + " & ".join(num(b) for b in bs) + " \\\\", "\\midrule",
     "Stock mode (default config) & " + " & ".join(pct(S[f"tune.q25.b{b}.stock_loss"]) for b in bs) + " \\\\",
     "Tuned table & " + " & ".join(pct(S[f"tune.q25.b{b}.tuned_loss"]) for b in bs) + " \\\\",
     "\\bottomrule", "\\end{tabular}"]
(OUT / "tab_tune.tex").write_text("\n".join(t) + "\n")
print("wrote tab_tune.tex")
