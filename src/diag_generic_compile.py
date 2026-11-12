#!/usr/bin/env python3
"""Diagnostic: is the generic default (src/defaulthook) applied inside torch.compile? Times one untuned Qwen2.5-7B shape at
M=32 four ways: default config eager, generic eager, generic under torch.compile, and the hook's selection logic traced."""
import os, json
import torch, triton.testing
import vllm.model_executor.determinism.batch_invariant as bi
import vllm.model_executor.determinism.batch_invariant_configs as bc

bc.resolve_tuned_matmul_configs()
n, k, m = 4608, 3584, 32
w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * k ** -0.5
x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
f = lambda: bi.linear_batch_invariant(x, w)
hook_active = bi._get_matmul_config.__name__ == "_get_matmul_config" and bi._get_matmul_config.__module__ == "sitecustomize"
print("hook active:", os.environ.get("VBI_GENERIC") is not None, "wrapped fn module:", bi._get_matmul_config.__module__)
eager = triton.testing.do_bench(f, warmup=20, rep=100)
cf = torch.compile(lambda a, b: bi.linear_batch_invariant(a, b), fullgraph=False)
cf(x, w)
compiled = triton.testing.do_bench(lambda: cf(x, w), warmup=20, rep=100)
d = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 3}
print("eager config:", bi._get_matmul_config(m, n, k, torch.bfloat16, d))
print(json.dumps({"eager_us": eager * 1e3, "compiled_us": compiled * 1e3}))
