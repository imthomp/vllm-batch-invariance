# vllm-batch-invariance

Does a request get the same output from vLLM no matter which other requests share its batch, or whether its prefix was cached? This repo measures that, measures what vLLM's batch-invariant mode costs to fix it, and tests a cheaper way to get the same guarantee on Hopper GPUs.

## Tested versions
Batch invariance is under active development in vLLM, and later versions may change both the cost and the routing described here. Results apply only to:

| | Version |
|---|---|
| vLLM (main) | `0.33.0rc1.dev454+gb8c04dd3b` (nightly, commit `b8c04dd3b`) |
| vLLM (release) | `0.32.0` |
| PyTorch | `2.14.0+cu130` with main; `2.13` with `0.32.0` |
| GPU | NVIDIA H100 80GB HBM3, driver `580.196.07` |

## Findings

**1. Without batch-invariant mode, outputs depend on serving state.** The same greedy request, run under cold or warm prefix cache and alone or alongside unrelated requests, produced different tokens on {{cross.main_qwen25_off.mismatch_requests}} of {{cross.main_qwen25_off.requests}} requests for Qwen2.5-7B-Instruct and {{cross.v032_qwen3_off.mismatch_requests}} of {{cross.v032_qwen3_off.requests}} for Qwen3-8B. Top-k logprobs differed on {{cross.main_qwen25_off.topk_diff_requests}} and {{cross.v032_qwen3_off.topk_diff_requests}} requests respectively.

**2. Stock batch-invariant mode fixes this but is expensive on models without tuned kernels.** Decode throughput loss at batch 32 on H100: {{main.qwen25.b32.stock_loss:pct}} for Qwen2.5-7B-Instruct on vLLM main, and {{v032.qwen3.b32.stock_loss:pct}} for Qwen3-8B on vLLM 0.32, before Qwen3-8B got tuned kernel configs.

**3. On Hopper, plain cuBLAS for large Linear layers gives the same determinism at almost no cost.** On SM90/SM100, vLLM's batch-invariant mode routes every Linear layer through a Triton kernel; this was a deliberate change in vLLM PR #35404, after GPT-OSS failed batch invariance with cuBLAS on H100. We find cuBLAS batch-invariant under the mode's settings for the large Linear shapes we tested, though a small mixture-of-experts router shape (N=32, K=2880) was reported not to be (vLLM issue #65569), so routing must be decided per shape. Routing Linear layers to `torch.nn.functional.linear` instead, with the rest of batch-invariant mode unchanged (`src/hopper_wrap.py`):

| Model (vLLM main) | Token mismatches | Requests with differing top-k | Loss at batch 32, stock | Loss at batch 32, plain cuBLAS |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct (no tuned config) | {{cross.main_qwen25_plain.mismatch_pairs}} / {{cross.main_qwen25_plain.pairs:int}} | {{cross.main_qwen25_plain.topk_diff_requests}} / {{cross.main_qwen25_plain.requests}} | {{main.qwen25.b32.stock_loss:pct}} | {{main.qwen25.b32.plain_loss:pct}} |
| Qwen3-8B (tuned config) | {{cross.main_qwen3_plain.mismatch_pairs}} / {{cross.main_qwen3_plain.pairs:int}} | {{cross.main_qwen3_plain.topk_diff_requests}} / {{cross.main_qwen3_plain.requests}} | {{main.qwen3.b32.stock_loss:pct}} | {{main.qwen3.b32.plain_loss:pct}} |
| Llama-3.1-8B-Instruct (tuned config) | {{cross.main_llama31_plain.mismatch_pairs}} / {{cross.main_llama31_plain.pairs:int}} | {{cross.main_llama31_plain.topk_diff_requests}} / {{cross.main_llama31_plain.requests}} | {{main.llama31.b32.stock_loss:pct}} | {{main.llama31.b32.plain_loss:pct}} |

Mismatches count every pair among 8 runs of each request: cold or warm cache, alone or with 7 unrelated neighbors, each twice. The same holds on vLLM 0.32 for Qwen3-8B and Llama-3.1-8B ({{cross.v032_qwen3_plain.mismatch_pairs}} and {{cross.v032_llama31_plain.mismatch_pairs}} mismatches). For the two models with tuned kernel configs on main, the gain over stock is small; the benefit is for models outside vLLM's tuned table.

**4. Why it works, and the main risk.** A direct test (`src/bitwise.py`) compares a row multiplied alone against the same row inside batches of 1 to 512 rows at {{bitwise.torch_default_flags.shapes}} model-sized shapes, on the same H100 with torch 2.13. Under the cuBLAS flags batch-invariant mode sets: {{bitwise.bi_flags_ws16.bad}} of {{bitwise.bi_flags_ws16.rows:int}} rows differ. Under PyTorch's default flags: {{bitwise.torch_default_flags.bad:int}} of {{bitwise.torch_default_flags.rows:int}} differ, all in {{bitwise.torch_default_flags.bad_shapes}} shape and starting at batch {{bitwise.torch_default_flags.min_bad_m}}. So the result depends on cuBLAS behavior that NVIDIA does not document as batch-invariant, and a future cuBLAS release could change it.

## Robustness and generalization (Phases 1 and 2)
Rules for both phases were fixed before running; see `PROTOCOL.md`, including every amendment.

**Is cuBLAS batch-invariant?** Each row multiplied alone was compared with the same row inside batches of 1 to 8192 rows, at every Linear shape of six models, in bf16 and fp16. Under batch-invariant mode's flags: {{paper.p1_bi_bad_total}} mismatches out of {{paper.p1_bi_comparisons:int}} comparisons across {{paper.p1_builds_h100}} H100 builds and {{paper.p1_builds_b200}} B200 builds ({{paper.p1_cublas_versions}} cuBLAS versions), including with a second process loading the GPU. For these large shapes the cuBLAS environment variables made no difference. For narrower shapes they do (see below), so they must be in effect before cuBLAS starts. With PyTorch defaults, {{paper.p1_ctl_min_h100:int}} to {{paper.p1_ctl_max_h100:int}} rows per run differed on H100 and {{paper.p1_ctl_max_b200:int}} on B200.

**Which shapes are batch-invariant?** A denser test (every batch size 1–1024, then every 16th to 8192; {{paper.p1c_shapes}} shapes including MoE routers and a sweep of narrow widths) on H100, at K = 2880 and 4096: every shape with N ≤ {{paper.p1c_max_fail_n}} is batch-variant under the flags, and every shape with N ≥ {{paper.p1c_min_pass_n}} is invariant at every batch size, in bf16 and fp16. A finer sweep shows the boundary also depends on K and on the cuBLAS workspace environment variables. With them, widths {{paper.p1d_narrow_ns:list}} pass at K = {{paper.p1d_narrow_pass_k:list}} but fail at {{paper.p1d_narrow_fail_k:list}}; widths from {{paper.p1d_mid_lo}} to {{paper.p1d_mid_hi}} pass only when they are multiples of {{paper.p1d_mid_pass_multiple}}; every width of at least {{paper.p1c_min_pass_n}} tested passed. Without them, every tested width from {{paper.p1d_noenv_min_fail_n}} to {{paper.p1d_noenv_max_fail_n}} fails at K = {{paper.p1d_noenv_fail_k:list}}, and only widths of {{paper.p1d_noenv_min_pass_n}} or more pass at every K. So the safe rule is a bitwise check of each shape, with N ≥ {{paper.p1c_min_pass_n}} as a conservative pre-filter. The gpt-oss-20b router (N=32, K=2880) diverges from batch size {{paper.p1c_router_bf16_first_bad_m}} in bf16. So routing must be per shape: wide projections (nearly all the cost) can use cuBLAS; routers keep the Triton kernel. On Blackwell, cuBLAS 13.4 documents an experimental batch-invariance mode (compute capability 10.x only), the natural long-term path there.

**Serving determinism with cuBLAS routing:** {{paper.p2_pass_configs}} of {{paper.p2_configs}} configurations gave identical tokens and top-k logprobs on every run of every request. That covers Qwen2.5-7B at TP=2 and TP=4, long context with chunked prefill, seeded sampling, Qwen2.5-32B at TP=2 and Mistral-7B. Controls with batch-invariant mode off diverged on {{paper.p2_ctl_div_min:pct}} to {{paper.p2_ctl_div_max:pct}} of requests.

**Gemma-2-9B fails, and it is vLLM's batch-invariant mode, not the routing:** stock mode diverged on {{phase2.cross.gemma2_on.mismatch_requests}} of {{phase2.cross.gemma2_on.requests}} requests, cuBLAS routing on {{phase2.cross.gemma2v2_plain.mismatch_requests}}. Diagnostics on {{phase2.cross.g2tri_on.requests}} requests each localize it: Triton attention gives {{phase2.cross.g2tri_on.mismatch_requests}} diverging requests and eager mode {{phase2.cross.g2eager_on.mismatch_requests}}, while removing softcapping still gives {{phase2.cross.g2nosc_on.mismatch_requests}}. Disabling the sliding window gives {{phase2.cross.g2nosw_on.mismatch_requests}}, and so does turning off CUDA graphs with compilation kept ({{phase2.cross.g2nocg_on.mismatch_requests}}), so the failure is in FlashAttention's sliding-window path under CUDA graphs.

**Cost at batch-32 decode, median of 3 jobs, vs. batch-invariant mode off:**

| Model | Stock | cuBLAS routing |
|---|---|---|
| Qwen2.5-7B | {{phase2.decode.q25.32.on_loss_median:pct}} | {{phase2.decode.q25.32.plain_loss_median:pct}} |
| Gemma-2-9B | {{phase2.decode.gemma2.32.on_loss_median:pct}} | {{phase2.decode.gemma2.32.plain_loss_median:pct}} |
| Mistral-7B (shapes mostly tuned) | {{phase2.decode.mistral.32.on_loss_median:pct}} | {{phase2.decode.mistral.32.plain_loss_median:pct}} |
| Qwen2.5-7B, TP=2 | {{phase2.decode.q25tp2.32.on_loss_median:pct}} | {{phase2.decode.q25tp2.32.plain_loss_median:pct}} |
| Qwen2.5-32B, TP=2 | {{phase2.decode.q32tp2.32.on_loss_median:pct}} | {{phase2.decode.q32tp2.32.plain_loss_median:pct}} |

Prefill: cuBLAS routing loses at most {{paper.p2_prefill_plain_max:pct}}, against {{paper.p2_prefill_stock_min:pct}} to {{paper.p2_prefill_stock_max:pct}} for stock mode. {{paper.p2_cost_pass}} of {{paper.p2_cost_cells}} cells meet the preregistered 5% limit. The exception is Qwen2.5-7B at TP=2. Batch-invariant mode's fixed-order communication settings alone cost {{phase2.decode.q25tp2.32.comm_loss_median:pct}} there, so the excess is communication, not the routing.

## Scope and limitations
- H100 80GB (B200 for the cuBLAS test only), tensor parallel up to 4, bf16, math prompts (GSM8K test and MATH-500).
- vLLM 0.32.0 (torch 2.13) and a vLLM main nightly, 0.33.0rc1.dev454 (torch 2.14, CUDA 13.0).
- Throughput: each cell is the median of 3 timed repeats in one job.
- Not tested: H200, multi-node, TP=8, mixture-of-experts or quantized models, production traffic, other engines' deterministic modes.
- fp32: cuBLAS fp32 matrix multiplication is not batch-invariant in our tests (raw results in `results/phase1b`), so the routing applies to bf16 and fp16 only. Bias terms in bf16/fp16 are fine.
- This is a measurement and a prototype patch (`src/plainpatch/`), not an upstream change.
- `paper/` holds a short technical note built from the same statistics.

## Prior art
Batch-dependent nondeterminism and batch-invariant kernels are known (He and Thinking Machines Lab, "Defeating Nondeterminism in LLM Inference", 2025; vLLM's batch-invariance feature). The contribution here is the Hopper routing measurement.

## Reproduce
```
python src/fetch_data.py problems.json              # needs `datasets`
# throughput: MODE off|on|plain
VLLM_BATCH_INVARIANT=1 python src/hopper_wrap.py src/decode_throughput.py --model Qwen/Qwen2.5-7B-Instruct --out out.json
# determinism
VLLM_BATCH_INVARIANT=1 python src/hopper_wrap.py src/crossover.py --data problems.json --out x.jsonl --model Qwen/Qwen2.5-7B-Instruct --n 100
python src/bitwise.py --workspace :16:8 --out bit.json
python src/analyze.py && python src/render.py && python tests/check.py
```
Set `VLLM_ENABLE_V1_MULTIPROCESSING=0` so the in-process patch reaches the engine. A patched run is valid only if its log contains `patched linear_batch_invariant`. `slurm/run.sbatch` is the job template used. `PROTOCOL.md` has the decision rules, fixed before each run.

Every number in this README is rendered from `results/stats.json`, which `src/analyze.py` recomputes from the raw files in `results/`.
