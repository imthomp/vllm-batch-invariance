#!/usr/bin/env python3
"""Phase 1: is cuBLAS matmul bitwise independent of batch size and row position?

A row multiplied alone is compared with the same row inside a batch of M rows, at rows near the start and the end
of the batch, for every Linear shape of the tested models, M from 1 to 8192, bf16 and fp16.
Modes:
  bi        env vars and torch flags exactly as vLLM batch-invariant mode sets them on SM90/SM100
  bi_noenv  torch flags only, as if the env vars were set too late to reach cuBLAS
  default   torch defaults, the negative control
--stress runs a second process doing large matmuls on the same GPU during the test.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["bi", "bi_noenv", "default"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--stress", action="store_true")
ap.add_argument("--trials", type=int, default=2)
ap.add_argument("--bias", action="store_true", help="add a bias vector, as in Linear layers with bias")
ap.add_argument("--dtypes", default="bfloat16,float16", help="comma-separated torch dtypes, e.g. float32")
ap.add_argument("--shapes", default="", help="semicolon-separated NxK list replacing the model shapes, e.g. 32x2880;128x2048")
ap.add_argument("--catch", action="store_true", help="record a RuntimeError per shape instead of stopping")
ap.add_argument("--dense", action="store_true", help="batch sizes 1..1024 then every 16th to 8192; also compare 4 middle rows")
a = ap.parse_args()
if a.mode == "bi":
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["CUBLASLT_WORKSPACE_SIZE"] = "1"
import torch
import torch.nn.functional as F

if a.mode != "default":
    # cuBLASLt first: some builds reject the split-K flag unless cuBLASLt is the backend.
    torch.backends.cuda.preferred_blas_library(backend="cublaslt")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (False, False)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
    torch.backends.cuda.matmul.fp32_precision = "ieee"

# (N, K) of each Linear weight, TP=1 unless noted.
SHAPES = {
    "qwen25_7b": [(4608, 3584), (3584, 3584), (37888, 3584), (3584, 18944), (152064, 3584)],
    "qwen3_8b": [(6144, 4096), (4096, 4096), (24576, 4096), (4096, 12288), (151936, 4096)],
    "llama31_8b": [(28672, 4096), (4096, 14336), (128256, 4096)],
    "mistral_7b": [(32768, 4096)],
    "gemma2_9b": [(8192, 3584), (3584, 4096), (28672, 3584), (3584, 14336), (256000, 3584)],
    "qwen25_32b_tp2": [(3584, 5120), (5120, 2560), (27648, 5120), (5120, 13824), (76032, 5120)],
}
MS = [1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512,
      1000, 1024, 2047, 2048, 4096, 8192]
if a.dense:
    MS = list(range(1, 1025)) + list(range(1040, 8193, 16))
if a.shapes:
    SHAPES = {"custom": [tuple(int(v) for v in sh.split("x")) for sh in a.shapes.split(";")]}
MAXM = max(MS)

stress = None
if a.stress:
    code = ("import torch\nx=torch.randn(8192,8192,device='cuda',dtype=torch.bfloat16)\n"
            "while True:\n    y=x@x\n    torch.cuda.synchronize()\n")
    stress = subprocess.Popen([sys.executable, "-c", code])
    time.sleep(20)

dev = torch.device("cuda")
torch.ones(8, 8, device=dev) @ torch.ones(8, 8, device=dev)


def cublaslt_version():
    import ctypes
    libs = {line.split()[-1] for line in open("/proc/self/maps") if "libcublasLt" in line}
    return [(p.rsplit("/", 1)[-1], ctypes.CDLL(p).cublasLtGetVersion()) for p in sorted(libs)]


CUBLASLT = cublaslt_version()
if os.environ.get("VBI_EXPECT_CUBLASLT"):  # e.g. 130802: abort unless that is the only cuBLASLt loaded
    assert {v for _, v in CUBLASLT} == {int(os.environ["VBI_EXPECT_CUBLASLT"])}, f"cuBLASLt mismatch: {CUBLASLT}"
print("torch", torch.__version__, "cuda", torch.version.cuda, "cublasLt", CUBLASLT,
      "gpu", torch.cuda.get_device_name(), "mode", a.mode, "stress", a.stress,
      "CUBLAS_WORKSPACE_CONFIG", os.environ.get("CUBLAS_WORKSPACE_CONFIG"), flush=True)
g = torch.Generator(device=dev).manual_seed(0)
rows = []
errors = []
t0 = time.time()
for dtype in [getattr(torch, d) for d in a.dtypes.split(",")]:
    for model, shapes in SHAPES.items():
        for n, k in shapes:
          try:
            w = (torch.randn(n, k, device=dev, dtype=dtype, generator=g) * k ** -0.5).to(dtype)
            b = torch.randn(n, device=dev, dtype=dtype, generator=g) if a.bias else None
            for trial in range(a.trials):
                x = torch.randn(MAXM, k, device=dev, dtype=dtype, generator=g)
                alone = {}
                for m in MS:
                    idx = set(range(min(m, 8))) | set(range(max(0, m - 8), m))
                    if a.dense:
                        idx |= {i for i in (m // 5, 2 * m // 5, 3 * m // 5, 4 * m // 5) if i < m}
                    idx = sorted(idx)
                    for i in idx:
                        if i not in alone:
                            alone[i] = F.linear(x[i:i + 1], w, b)
                    out = F.linear(x[:m], w, b)
                    bad = sum(bool((out[i] != alone[i][0]).any()) for i in idx)
                    rows.append({"dtype": str(dtype).split(".")[-1], "model": model, "n": n, "k": k,
                                 "trial": trial, "m": m, "rows": len(idx), "bad": bad})
                del x, alone
            del w
          except RuntimeError as e:
            if not a.catch:
                raise
            errors.append({"dtype": str(dtype).split(".")[-1], "n": n, "k": k, "error": str(e)[:300]})
            print("ERROR", n, k, dtype, str(e)[:120], flush=True)
          torch.cuda.empty_cache()
        print(model, str(dtype), "bad so far", sum(r["bad"] for r in rows), f"{time.time() - t0:.0f}s", flush=True)
if stress:
    stress.kill()
tot = sum(r["bad"] for r in rows)
print("SUMMARY mode", a.mode, "stress", a.stress, "comparisons", sum(r["rows"] for r in rows), "bad", tot, "errors", len(errors),
      "bad cells", sorted({(r["dtype"], r["n"], r["k"], r["m"]) for r in rows if r["bad"]})[:40], flush=True)
json.dump({"torch": torch.__version__, "cuda": torch.version.cuda, "cublaslt": CUBLASLT, "gpu": torch.cuda.get_device_name(),
           "mode": a.mode, "stress": a.stress, "bias": a.bias, "dtypes": a.dtypes, "rows": rows, "errors": errors,
           "cublas_bi_flags": os.environ.get("CUBLAS_BATCH_INVARIANCE_FLAGS")}, open(a.out, "w"))
