#!/usr/bin/env python3
"""Is cuBLAS matmul bitwise independent of batch shape? Row alone against the same row embedded in a batch."""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--workspace", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--plain", action="store_true")
a = ap.parse_args()
if a.workspace:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = a.workspace
    os.environ["CUBLASLT_WORKSPACE_SIZE"] = "1"
import torch
import torch.nn.functional as F

if not a.plain:
    torch.backends.cuda.preferred_blas_library(backend="cublaslt")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
g = torch.Generator("cuda").manual_seed(0)
shapes = [(4096, 4096), (4096, 22016), (4096, 12288), (4096, 128256), (3072, 8192), (1024, 4096), (768, 32000), (256, 256)]
Ms = [1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512]
rows = []
for k, n in shapes:
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g)
    for trial in range(5):
        full = torch.randn(512, k, device="cuda", dtype=torch.bfloat16, generator=g)
        alone = torch.cat([F.linear(full[i:i + 1], w) for i in range(16)]).float()
        for m in Ms:
            sub = full[:m]
            out = F.linear(sub, w).float()
            q = min(m, 16)
            bad = (out[:q] != alone[:q]).any(-1).sum().item()
            rows.append({"k": k, "n": n, "trial": trial, "m": m, "rows": q, "bad": bad})
tot = sum(r["bad"] for r in rows)
print("plain", a.plain, "workspace", repr(a.workspace), "comparisons", sum(r["rows"] for r in rows), "bad rows", tot,
      "bad M", sorted({r["m"] for r in rows if r["bad"]}), "bad shapes", sorted({(r["k"], r["n"]) for r in rows if r["bad"]}))
json.dump(rows, open(a.out, "w"))
