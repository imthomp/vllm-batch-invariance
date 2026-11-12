#!/usr/bin/env python3
"""Diagnostic: at large batch sizes (beyond the tuned 2048 bucket), is the tuned config slower than vLLM's default?
For each gpt-oss shape, time the tuned table entry's last bucket and the default config at M = 1024..16384."""
import json, sys
import torch, triton.testing
import vllm.model_executor.determinism.batch_invariant as bi
orig = bi._get_matmul_config
dft = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 3}
ents = [e for f in sys.argv[1:] for e in json.load(open(f))["entries"]]
print(torch.cuda.get_device_name())
for e in ents:
    n, k = e["n"], e["k"]
    b = e["m_buckets"][-1]
    tuned = {"BLOCK_SIZE_M": b[1], "BLOCK_SIZE_N": b[2], "BLOCK_SIZE_K": e["block_k"], "GROUP_SIZE_M": 8, "num_warps": b[3], "num_stages": b[4]}
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * k ** -0.5
    out = []
    for m in (1024, 2048, 4096, 8192, 16384):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        t = {}
        for name, c in (("tuned", tuned), ("default", dft)):
            bi._get_matmul_config = lambda M, N, K, d, dd, c=c: c
            t[name] = triton.testing.do_bench(lambda: bi.linear_batch_invariant(x, w), warmup=10, rep=40)
        out.append(f"M{m}: tuned/default {t['tuned'] / t['default']:.2f}")
        del x
    bi._get_matmul_config = orig
    print(f"{n}x{k} last bucket {b}  " + "  ".join(out), flush=True)
