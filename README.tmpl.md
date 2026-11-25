# vllm-batch-invariance

[![checks](https://github.com/imthomp/vllm-batch-invariance/actions/workflows/checks.yml/badge.svg)](https://github.com/imthomp/vllm-batch-invariance/actions/workflows/checks.yml)

The same request can get a different output from vLLM depending on which other requests share its batch, or whether its prefix was cached. That makes individual evaluation records depend on serving load. vLLM's batch-invariant mode removes the dependence, but on models without tuned kernels it costs most of the decode throughput. This repo measures the effect and the cost, finds where the cost comes from, and reduces it. The work fed upstream vLLM changes and a paper draft.

## TL;DR

- Serving state can change greedy outputs when batch-invariant mode is disabled.
- Stock batch-invariant mode can impose a large decode-throughput cost on model
  shapes without tuned kernels.
- A generic Hopper fallback preserved invariance in the recorded checks while
  recovering {{paper.generic_recovered_b32:pct}} of Qwen2.5-7B's batch-32
  tuning gap; the corresponding upstream change is under review.
- Tuned gpt-oss H100 configurations and a Hopper/Blackwell correctness comment
  have already merged into vLLM.

## Status
Upstream vLLM contributions:

| | What | State |
|---|---|---|
| [#65931](https://github.com/vllm-project/vllm/pull/65931) | Tuned H100 batch-invariant matmul configs for gpt-oss | merged |
| [#67179](https://github.com/vllm-project/vllm/pull/67179) | Correct the batch-invariance comment on Hopper/Blackwell split-k | merged |
| [#67376](https://github.com/vllm-project/vllm/pull/67376) | Generic per-batch-size Hopper fallback config for untuned bf16 shapes | under review |
| [#68467](https://github.com/vllm-project/vllm/pull/68467) | Breakable CUDA graphs for MXFP4 models (gpt-oss) | open |
| [#67275](https://github.com/vllm-project/vllm/issues/67275) | RFC: batch-size-aware configs for untuned shapes | open |
| [#65587](https://github.com/vllm-project/vllm/issues/65587) | Bug report: Gemma-2-9B not batch-invariant on H100; fixed by another contributor in [#65627](https://github.com/vllm-project/vllm/pull/65627) | fixed |
| [#65569](https://github.com/vllm-project/vllm/issues/65569) | Proposal to route Linear layers to cuBLAS on Hopper | closed by us (see finding 6) |

- **Paper:** `paper/main.tex` (named version) and `paper/acl.tex` (anonymous ARR review version) share their content in `paper/body.tex` and `paper/appendix.tex`. The working sources and PDF are visible in this repository, but the paper has not yet been formally released or submitted; it goes to a lab workshop first.
- **Evaluation experiment:** pre-registered and scored (finding 8).

## Findings
Every number below is rendered from `results/stats.json`. Decision rules were fixed before each run and are in `PROTOCOL.md`, with every amendment and void run.

**1. Without batch-invariant mode, outputs depend on serving state.** The same greedy request, run under cold or warm prefix cache and alone or alongside unrelated requests, produced different tokens on {{cross.main_qwen25_off.mismatch_requests}} of {{cross.main_qwen25_off.requests}} requests for Qwen2.5-7B-Instruct and {{cross.v032_qwen3_off.mismatch_requests}} of {{cross.v032_qwen3_off.requests}} for Qwen3-8B. In a WMT23 machine-translation pilot, {{paper.mt_change_min:pct}} to {{paper.mt_change_max:pct}} of segments changed across batching configurations, but {{paper.mt_flips}} of {{paper.mt_comparisons}} system comparisons changed.

**2. Stock batch-invariant mode is expensive on models without tuned kernels.** Decode throughput loss at batch 32 on H100, against the mode off: {{phase2.decode.q25.32.on_loss_median:pct}} for Qwen2.5-7B-Instruct and {{phase2.decode.gemma2.32.on_loss_median:pct}} for Gemma-2-9B, but {{main.llama31.b32.stock_loss:pct}} for Llama-3.1-8B and {{main.qwen3.b32.stock_loss:pct}} for Qwen3-8B, whose shapes have tuned entries in vLLM's config table.

**3. The cost is a tuning gap.** Shapes without a table entry use one default configuration with 128-row tiles at every batch size, so at decode batch sizes most of each tile is padding. Tuning the five Qwen2.5-7B shapes per batch-size bucket cut the loss from {{paper.tune_stock_min:pct}}–{{paper.tune_stock_max:pct}} to {{paper.tune_loss_min:pct}}–{{paper.tune_loss_max:pct}} across batch sizes up to {{paper.tune_bmax}} ({{generic2.q25gen2.b32.tuned_loss:pct}} at batch 32), with {{tune.q25.check_bad}} of {{tune.q25.check_rows:int}} rows differing in a bitwise alone-vs-batched check. Tuned gpt-oss configs were merged upstream (#65931); they raised decode-heavy serving throughput by {{serve_gptoss.decode.gain_over_stock:pct}} for gpt-oss-20b and {{serve_gptoss120b.decode.gain_over_stock:pct}} for gpt-oss-120b, which is small because most of gpt-oss runs in quantized expert kernels the table does not touch.

**4. One generic configuration recovers most of the tuning gain without per-model work.** A per-batch-size configuration chosen on training shapes that exclude both evaluation models reduces Qwen2.5-7B's batch-32 loss from {{paper.generic_stock_b32:pct}} to {{paper.generic_b32:pct}} (per-model tuning: {{paper.generic_tuned_b32:pct}}), recovering {{paper.generic_recovered_b32:pct}} of the gap, and Gemma-2-9B's from {{paper.generic_gemma_stock_b32:pct}} to {{paper.generic_gemma_b32:pct}}. Invariance is preserved: {{paper.generic_check_bad}} of {{paper.generic_check_rows:int}} rows differ in the bitwise check, and {{paper.generic_cross_mismatch}} of {{paper.generic_cross_requests}} requests change across serving conditions. Served from the upstream implementation (#67376) against its merge base, it raises concurrency-32 decode throughput by {{paper.core_untuned_decode32_min:pct}}–{{paper.core_untuned_decode32_max:pct}} on every untuned model tested (Qwen2.5-7B on H100 and H200, Qwen2.5-32B at TP=2, Gemma-2-9B) and Qwen2.5-7B single-request decode by {{paper.core_q25_decode1:pct}}. Mistral-7B, whose shapes are mostly tuned already, changes by {{paper.core_mistral_decode32:pct}}.

**5. Breakable CUDA graphs are needed, and they have a prefill cost on some models.** The config lookup depends on the runtime batch size, and a compiled forward can freeze the lookup made during graph capture. Breakable graphs keep it dynamic. vLLM enables them automatically only for unquantized models; letting MXFP4 models (gpt-oss) through (#68467) raises gpt-oss-20b single-request decode by {{paper.ossgate_decode1_min:pct}}–{{paper.ossgate_decode1_max:pct}} and concurrency-32 decode by {{paper.ossgate_decode32_min:pct}}–{{paper.ossgate_decode32_max:pct}} on H100 and H200, with prefill within {{paper.ossgate_prefill32_absmax:pct}}. On Gemma-2-9B, #67376 slows prefill by {{paper.core_gemma_prefill32_loss:pct}}. A diagnostic run traces this to the graph mode, not the configuration: forcing breakable graphs on the merge base changes prefill throughput by {{paper.gemmadiag_basebrk_prefill32:pct}}, while the PR's configuration without breakable graphs changes it by {{paper.gemmadiag_prnobrk_prefill32:pct}}. Extra batch-size buckets for the gpt-oss routers were faster at the kernel level but not end to end (gpt-oss-120b prefill {{followup.h100_oss120_bk.prefill32.gatebk_vs_gate_median:pct}}, decode {{followup.h100_oss120_bk.decode32.gatebk_vs_gate_median:pct}}, {{followup.h100_oss120_bk.rounds}} rounds), so they were not proposed.

**6. Routing Linear layers to cuBLAS is cheapest but not recommended.** Under the mode's settings, cuBLAS gave identical outputs for large shapes, and routing Linear layers to it cut Qwen2.5-7B's batch-32 loss to {{phase2.decode.q25.32.plain_loss_median:pct}}. This was the project's first proposal (#65569). It was withdrawn after maintainer feedback, because the property is undocumented and shape-dependent: on H100, every shape with N ≤ {{paper.p1c_max_fail_n}} is batch-variant and every shape with N ≥ {{paper.p1c_min_pass_n}} invariant, at K = 2880 and 4096; the boundary moves with K, with the cuBLAS environment variables and across GPUs, and the gpt-oss-20b router (N=32, K=2880) diverges from batch size {{paper.p1c_router_bf16_first_bad_m}}. On Blackwell, cuBLAS's documented batch-invariance mode passes the bitwise test unless vLLM's reduced-precision setting is also applied. Raw results are in `results/phase1*`.

**7. Gemma-2-9B was not batch-invariant under vLLM's own mode.** Stock mode diverged on {{phase2.cross.gemma2_on.mismatch_requests}} of {{phase2.cross.gemma2_off.requests}} requests on H100. One-component-at-a-time diagnostics localized it to FlashAttention's sliding-window path under CUDA graphs (#65587). The upstream fix (#65627) takes the issue's reproducer from {{paper.fafix_base_div}} of {{paper.fafix_n}} prompts differing to {{paper.fafix_fix_div}}.

**8. Serving changes evaluation records, not (so far) evaluation conclusions.** Six instruction-tuned models (Qwen2.5-7B, Llama-3.1-8B, Gemma-2-9B, Mistral-7B, OLMo-2-7B, Phi-4-mini) on GSM8K and MATH-500, under four serving conditions (serial, full batch, mixed with unrelated prompts, chunked prefill), each twice, with batch-invariant mode off and on, on H100. Between conditions, {{paper.ev_flip_between_min:pct}} to {{paper.ev_flip_between_max:pct}} of examples changed correctness (mean per model and benchmark; at most {{paper.ev_flip_within_max:pct}} between identical repeats), and accuracy spanned at most {{paper.ev_spread_max:pct}}. But {{evalvar.comparisons_changed_condition_only}} of {{evalvar.comparisons}} pairwise model comparisons changed sign or significance beyond repeat noise, and the model ranking was the same in every run. With the mode on, {{paper.ev_bi_differ_total}} requests differed across conditions for any model; generation took {{paper.ev_time_C2_min:pct}} to {{paper.ev_time_C2_max:pct}} longer in the full-batch condition. Design and decision rule: `PROTOCOL.md`, "Do serving conditions change NLP evaluation conclusions?". Code: `src/evalvar_gen.py`, `slurm/evalvar.sbatch`, `src/evalvar_score.py`; runs in `results/validation/evalvar/`.

## Tested versions
Batch invariance is under active development in vLLM; later versions may change both the cost and the behavior described here.

| | Version |
|---|---|
| vLLM, cost and determinism studies | `0.33.0rc1.dev454+gb8c04dd3b` (main nightly) and `0.32.0` |
| vLLM, upstream-PR and follow-up experiments | runtime `0.34.0rc1.dev352`, with source trees on PR merge base `df8c0296` |
| PyTorch | `2.14.0+cu130` (main), `2.13` (`0.32.0`) |
| GPUs | H100 80GB HBM3 (most results); H200 and B200 where noted |

## Repository layout
- `PROTOCOL.md`: every pre-registered rule, outcome, amendment and void run, dated.
- `src/`: experiments (`crossover*.py` determinism, `decode_throughput.py` / `prefill_throughput.py` cost, `bitwise*.py` kernel checks, `tune_bi.py` / `tune_generic.py` configs, `evalvar_*.py` evaluation variance), scorers (`score_*.py`, `analyze.py`, `paper_stats.py`), and renderers (`render.py`, `make_macros.py`, `make_figures.py`).
- `slurm/`: job scripts. `serve_bench.sbatch`, `core_pr_serve.sbatch` and `oss_followup_serve.sbatch` run `vllm bench serve` comparisons; `evalvar.sbatch` runs the evaluation-variance design.
- `results/`: raw outputs and `stats.json`, the single source of every reported number. `results/validation/` holds the upstream-PR serving runs.
- `paper/`: LaTeX sources; `macros.tex` and `tab_*.tex` are generated from `stats.json`.
- `tests/`: `check.py` (stats and README are current, no hand-typed numbers) and `check_paper.py` (macros current, no hand-typed numbers, every table row traced to data, the ARR version anonymous).

## Reproduce
```
python src/fetch_data.py problems.json              # GSM8K test + MATH-500; needs `datasets`
# cost and determinism (wrap with src/hopper_wrap.py to route Linear layers to cuBLAS)
VLLM_BATCH_INVARIANT=1 python src/decode_throughput.py --model Qwen/Qwen2.5-7B-Instruct --out out.json
VLLM_BATCH_INVARIANT=1 python src/crossover.py --data problems.json --out x.jsonl --model Qwen/Qwen2.5-7B-Instruct --n 100
python src/bitwise.py --workspace :16:8 --out bit.json
# upstream PR vs merge base, online serving
sbatch --export=ALL,TAG=<tag>,MODEL=Qwen/Qwen2.5-7B-Instruct,TP=1,BASE_SRC=<merge-base tree>,PR_SRC=<PR tree> slurm/core_pr_serve.sbatch
# evaluation variance: one job per model and mode, then score
sbatch --export=ALL,MODEL=Qwen/Qwen2.5-7B-Instruct,SHORT=qwen25_7b,BI=1 slurm/evalvar.sbatch
python src/evalvar_archive.py && python src/evalvar_score.py --root results/validation/evalvar --merge
# numbers, README, paper
python src/analyze.py && python src/paper_stats.py && python src/render.py && python src/make_macros.py
python tests/check.py && python tests/check_paper.py
cd paper && latexmk -pdf main.tex && latexmk -pdf acl.tex
```
Source trees for the PR experiments need vLLM's compiled extensions, linked from a matching build. The job scripts record the commit of every tree they run. Job scripts use `HF_HUB_OFFLINE=1` and cluster paths, which you will need to adjust.

## Scope and limitations
- One engine (vLLM), bf16, dense models up to 32B plus gpt-oss, single node, tensor parallel up to 4. Not tested: multi-node, production traffic, other engines' deterministic modes.
- The generic configuration targets Hopper; on other GPUs vLLM's default applies.
- Throughput comparisons use three rotated rounds (five for the 120b router test) and report medians and per-round ranges, not confidence intervals.
- cuBLAS behavior was measured on the library versions listed and may change with any release.

## Prior art
Batch-dependent nondeterminism and batch-invariant kernels are known (He and Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference", 2025; vLLM's batch-invariance feature; SGLang's deterministic mode). Tuned configs for vLLM's batch-invariant kernel have been contributed for individual models (e.g. #62219). This repo measures how much of the cost tuning recovers, replaces per-model tuning with a generic default, and tests whether the remaining nondeterminism matters for evaluation.

Every number in this README is a placeholder in `README.tmpl.md`, filled from `results/stats.json` by `src/render.py`; `tests/check.py` fails if the README is stale or the template contains an unreviewed number.
