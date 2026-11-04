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

**1. Without batch-invariant mode, outputs depend on serving state.** The same greedy request, run under cold or warm prefix cache and alone or alongside unrelated requests, produced different tokens on 41 of 100 requests for Qwen2.5-7B-Instruct and 69 of 100 for Qwen3-8B. Top-k logprobs differed on 100 and 100 requests respectively.

**2. Stock batch-invariant mode fixes this but is expensive on models without tuned kernels.** Decode throughput loss at batch 32 on H100: 62.3% for Qwen2.5-7B-Instruct on vLLM main, and 61.4% for Qwen3-8B on vLLM 0.32, before Qwen3-8B got tuned kernel configs.

**3. On Hopper, plain cuBLAS for large Linear layers gives the same determinism at almost no cost.** On SM90/SM100, vLLM's batch-invariant mode routes every Linear layer through a Triton kernel; this was a deliberate change in vLLM PR #35404, after GPT-OSS failed batch invariance with cuBLAS on H100. We find cuBLAS batch-invariant under the mode's settings for the large Linear shapes we tested, though a small mixture-of-experts router shape (N=32, K=2880) was reported not to be (vLLM issue #65569), so routing must be decided per shape. Routing Linear layers to `torch.nn.functional.linear` instead, with the rest of batch-invariant mode unchanged (`src/hopper_wrap.py`):

| Model (vLLM main) | Token mismatches | Requests with differing top-k | Loss at batch 32, stock | Loss at batch 32, plain cuBLAS |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct (no tuned config) | 0 / 2,800 | 0 / 100 | 62.3% | 0.7% |
| Qwen3-8B (tuned config) | 0 / 2,800 | 0 / 100 | 9.1% | 4.2% |
| Llama-3.1-8B-Instruct (tuned config) | 0 / 2,800 | 0 / 100 | −0.5% | 1.0% |

Mismatches count every pair among 8 runs of each request: cold or warm cache, alone or with 7 unrelated neighbors, each twice. The same holds on vLLM 0.32 for Qwen3-8B and Llama-3.1-8B (0 and 0 mismatches). For the two models with tuned kernel configs on main, the gain over stock is small; the benefit is for models outside vLLM's tuned table.

**4. Why it works, and the main risk.** A direct test (`src/bitwise.py`) compares a row multiplied alone against the same row inside batches of 1 to 512 rows at 8 model-sized shapes, on the same H100 with torch 2.13. Under the cuBLAS flags batch-invariant mode sets: 0 of 12,200 rows differ. Under PyTorch's default flags: 1,200 of 12,200 differ, all in 1 shape and starting at batch 17. So the result depends on cuBLAS behavior that NVIDIA does not document as batch-invariant, and a future cuBLAS release could change it.

## Robustness and generalization (Phases 1 and 2)
Rules for both phases were fixed before running; see `PROTOCOL.md`, including every amendment.

**Is cuBLAS batch-invariant?** Each row multiplied alone was compared with the same row inside batches of 1 to 8192 rows, at every Linear shape of six models, in bf16 and fp16. Under batch-invariant mode's flags: 0 mismatches out of 769,920 comparisons across 7 H100 builds and 2 B200 builds (4 cuBLAS versions), including with a second process loading the GPU. The cuBLAS environment variables turned out not to matter; the PyTorch flags alone suffice. With PyTorch defaults, 6,784 to 8,544 rows per run differed on H100 and 19,544 on B200.

**Which shapes are batch-invariant?** A denser test (every batch size 1–1024, then every 16th to 8192; 37 shapes including MoE routers and a sweep of narrow widths) splits cleanly on H100: every shape with N ≤ 96 is batch-variant under the flags, and every shape with N ≥ 128 is invariant at every batch size, in bf16 and fp16. The gpt-oss-20b router (N=32, K=2880) diverges from batch size 23 in bf16. So routing must be per shape: wide projections (nearly all the cost) can use cuBLAS; routers keep the Triton kernel. On Blackwell, cuBLAS 13.4 documents an experimental batch-invariance mode (compute capability 10.x only), the natural long-term path there.

**Serving determinism with cuBLAS routing:** 6 of 7 configurations gave identical tokens and top-k logprobs on every run of every request. That covers Qwen2.5-7B at TP=2 and TP=4, long context with chunked prefill, seeded sampling, Qwen2.5-32B at TP=2 and Mistral-7B. Controls with batch-invariant mode off diverged on 4.0% to 76.0% of requests.

**Gemma-2-9B fails, and it is vLLM's batch-invariant mode, not the routing:** stock mode diverged on 40 of 100 requests, cuBLAS routing on 39. Diagnostics on 30 requests each localize it: Triton attention gives 0 diverging requests and eager mode 0, while removing softcapping still gives 15. Disabling the sliding window gives 0, and so does turning off CUDA graphs with compilation kept (0), so the failure is in FlashAttention's sliding-window path under CUDA graphs.

**Cost at batch-32 decode, median of 3 jobs, vs. batch-invariant mode off:**

| Model | Stock | cuBLAS routing |
|---|---|---|
| Qwen2.5-7B | 63.1% | 0.2% |
| Gemma-2-9B | 59.2% | 1.0% |
| Mistral-7B (shapes mostly tuned) | 5.8% | 3.3% |
| Qwen2.5-7B, TP=2 | 61.0% | 5.8% |
| Qwen2.5-32B, TP=2 | 58.2% | −2.3% |

Prefill: cuBLAS routing loses at most 5.0%, against 9.5% to 25.5% for stock mode. 16 of 17 cells meet the preregistered 5% limit. The exception is Qwen2.5-7B at TP=2. Batch-invariant mode's fixed-order communication settings alone cost 6.0% there, so the excess is communication, not the routing.

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
