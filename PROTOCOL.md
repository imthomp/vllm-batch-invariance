# Decision rules

Each rule was written down before the runs it governs. Outcomes are in `results/stats.json`.

## Determinism (both vLLM versions)
- 100 requests per model, 8 runs each (cold/warm prefix cache x alone/with 7 unrelated neighbors, twice), greedy, up to 256 new tokens.
- Pass: zero token mismatches across all pairs of runs of every request. Any mismatch fails.
- A control with batch-invariant mode off must diverge, or the test has no power.
- Warm runs must show prefix-cache hits; truncated runs are excluded.
- A patched run counts only if its log contains `patched linear_batch_invariant`.

## Cost on vLLM main
vLLM main includes H100 tuned matmul configs for Qwen3-8B and Llama-3.1-8B (vLLM PR #62219) and inductor batch-invariant mode (#61118). Qwen2.5-7B-Instruct has no tuned config.
- Pass: on the untuned model, plain cuBLAS loses at least 10 percentage points less decode throughput than stock batch-invariant mode at batch 32, and determinism passes with a diverging control.
- If the gap is under 10 points on all models: closed, tuning already fixes it.
- If it holds only on the untuned model: the claim is limited to models without tuned configs. This is the outcome observed.

## Bitwise matmul
Row alone vs the same row inside batches of 1 to 512 rows. Any mismatch under the batch-invariant cuBLAS flags fails the claim that split-k is the only source of batch variance. PyTorch's default flags serve as the negative control and are expected to mismatch.

## Phase 1: is the cuBLAS behavior reliable? (fixed 2026-11-03, before any Phase 1 data)
Test: `src/bitwise2.py`. Every Linear shape of Qwen2.5-7B, Qwen3-8B, Llama-3.1-8B, Mistral-7B, Gemma-2-9B and Qwen2.5-32B at TP=2; batch M from 1 to 8192; rows near the start and end of each batch; bf16 and fp16; two random draws.
Grid: torch 2.12+cu126, 2.13+cu128, 2.13+cu130, 2.14+cu126, 2.14+cu128, 2.14+cu130 on H100; one torch build on B200 and H200 where the queue allows; a run with a second process doing matmuls on the same GPU.
Modes: `bi` (env vars and flags as vLLM sets them), `bi_noenv` (flags only), `default` (negative control).
- Pass: zero mismatches in `bi` mode in every cell, on every GPU type and build tested.
- Power: `default` must mismatch somewhere on each GPU type. If it never does on a GPU type, that GPU's pass is reported as uninformative.
- Any `bi` mismatch: the claim that cuBLAS is batch-invariant under these settings fails for that build and GPU. The method then needs a startup self-test with fallback, and is rejected outright for a build where a mismatch occurs at a shape and batch size the tested models actually use.
- `bi_noenv` is descriptive: it shows whether the cuBLAS env vars matter, which decides whether their late setting in vLLM is a risk.

## Phase 2: does it generalize? (fixed 2026-11-03, before any Phase 2 data)
Patch: `src/plainpatch/sitecustomize.py`, which patches every process, TP workers included. A run counts only if its log has one marker per engine process: at least TP+1 markers, or more if vLLM starts more processes.
Determinism (`src/crossover2.py`, plain cuBLAS, batch-invariant on). Configurations:
(a) Qwen2.5-7B TP=2; (b) Qwen2.5-7B TP=4; (c) Qwen2.5-7B long context, 150-shot prefix of about 15k tokens, chunked prefill with a 512-token budget, 50 requests; (d) Qwen2.5-7B seeded sampling, temperature 0.7; (e) Mistral-7B; (f) Gemma-2-9B; (g) Qwen2.5-32B TP=2. 100 requests unless stated.
- Pass per configuration: zero token mismatches across all pairs of runs of every request.
- Controls with batch-invariant off for (a), (c), (d), (e), (f), (g) must diverge on at least one request. (b) shares (a)'s control. A configuration whose control does not diverge is reported as uninformative.
- Warm runs must reach a prefix-cache hit of at least 0.9.
Cost, three separate jobs per cell, median reported with the range:
- Decode (`src/decode_throughput.py`), batch 32: Qwen2.5-7B, Mistral-7B and Gemma-2-9B at TP=1; Qwen2.5-7B and Qwen2.5-32B at TP=2.
- Prefill (`src/prefill_throughput.py`), 1k and 4k prompt tokens, batch 1, 8 and 32: Qwen2.5-7B and Mistral-7B.
- Pass per cell: plain loses at most 5% throughput against batch-invariant off. For decode on untuned models, stock batch-invariant also loses at least 10 points more than plain.
- The README says "under 1%" for a cell only if all three repeats are under 1%.
Generalization claims are made only for the configurations that pass. Failures are reported in the README.
