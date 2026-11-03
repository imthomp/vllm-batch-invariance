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

**3. On Hopper, plain cuBLAS for Linear layers gives the same determinism at almost no cost.** On SM90/SM100, vLLM's batch-invariant mode already configures cuBLAS (workspace and cuBLASLt flags) as if split-k were the only source of batch variance, yet it still routes every Linear layer through a Triton kernel. Routing Linear layers to `torch.nn.functional.linear` instead, with the rest of batch-invariant mode unchanged (`src/hopper_wrap.py`):

| Model (vLLM main) | Token mismatches | Requests with differing top-k | Loss at batch 32, stock | Loss at batch 32, plain cuBLAS |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct (no tuned config) | {{cross.main_qwen25_plain.mismatch_pairs}} / {{cross.main_qwen25_plain.pairs:int}} | {{cross.main_qwen25_plain.topk_diff_requests}} / {{cross.main_qwen25_plain.requests}} | {{main.qwen25.b32.stock_loss:pct}} | {{main.qwen25.b32.plain_loss:pct}} |
| Qwen3-8B (tuned config) | {{cross.main_qwen3_plain.mismatch_pairs}} / {{cross.main_qwen3_plain.pairs:int}} | {{cross.main_qwen3_plain.topk_diff_requests}} / {{cross.main_qwen3_plain.requests}} | {{main.qwen3.b32.stock_loss:pct}} | {{main.qwen3.b32.plain_loss:pct}} |
| Llama-3.1-8B-Instruct (tuned config) | {{cross.main_llama31_plain.mismatch_pairs}} / {{cross.main_llama31_plain.pairs:int}} | {{cross.main_llama31_plain.topk_diff_requests}} / {{cross.main_llama31_plain.requests}} | {{main.llama31.b32.stock_loss:pct}} | {{main.llama31.b32.plain_loss:pct}} |

Mismatches count every pair among 8 runs of each request: cold or warm cache, alone or with 7 unrelated neighbors, each twice. The same holds on vLLM 0.32 for Qwen3-8B and Llama-3.1-8B ({{cross.v032_qwen3_plain.mismatch_pairs}} and {{cross.v032_llama31_plain.mismatch_pairs}} mismatches). For the two models with tuned kernel configs on main, the gain over stock is small; the benefit is for models outside vLLM's tuned table.

**4. Why it works, and the main risk.** A direct test (`src/bitwise.py`) compares a row multiplied alone against the same row inside batches of 1 to 512 rows at {{bitwise.torch_default_flags.shapes}} model-sized shapes, on the same H100 with torch 2.13. Under the cuBLAS flags batch-invariant mode sets: {{bitwise.bi_flags_ws16.bad}} of {{bitwise.bi_flags_ws16.rows:int}} rows differ. Under PyTorch's default flags: {{bitwise.torch_default_flags.bad:int}} of {{bitwise.torch_default_flags.rows:int}} differ, all in {{bitwise.torch_default_flags.bad_shapes}} shape and starting at batch {{bitwise.torch_default_flags.min_bad_m}}. So the result depends on cuBLAS behavior that NVIDIA does not document as batch-invariant, and a future cuBLAS release could change it.

## Scope and limitations
- One H100 80GB GPU, tensor parallel 1, bf16, greedy decoding, decode-heavy math prompts (GSM8K test and MATH-500).
- vLLM 0.32.0 (torch 2.13) and a vLLM main nightly, 0.33.0rc1.dev454 (torch 2.14, CUDA 13.0).
- Throughput: each cell is the median of 3 timed repeats in one job.
- Not tested: multi-GPU, H200 or Blackwell, prefill-heavy workloads, other cuBLAS versions, sampling.
- This is a measurement and a prototype patch, not an upstream change.

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
