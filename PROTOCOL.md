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
