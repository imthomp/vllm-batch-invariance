#!/usr/bin/env python3
"""Tune vLLM's batch-invariant Triton matmul for given Linear shapes, in vLLM's own tuned-table format.

vLLM's table (vllm/model_executor/determinism/batch_invariant_configs.py) holds, per (N, K), one BLOCK_K for the
whole shape and per batch-size bucket a (BLOCK_M, BLOCK_N, num_warps, num_stages). Only BLOCK_K fixes the
reduction order, so any table of this form stays batch-invariant; --check verifies that bitwise anyway.

Search: for each BLOCK_K candidate, the fastest config per bucket (timed at the bucket's largest batch size);
the BLOCK_K kept is the one with the lowest mean log time ratio to cuBLAS over all buckets.
Usage: tune_bi.py --shapes "4608x3584;..." --out tuned.json [--check]
"""
import argparse
import itertools
import json
import math
import time

import torch
import torch.nn.functional as F
import triton.testing

import vllm.model_executor.determinism.batch_invariant as bi
import vllm.model_executor.determinism.batch_invariant_configs as bc

ap = argparse.ArgumentParser()
ap.add_argument("--shapes", required=True, help="semicolon-separated NxK")
ap.add_argument("--out", required=True)
ap.add_argument("--quick", action="store_true", help="tiny search space and grid, for smoke tests only")
ap.add_argument("--check", action="store_true", help="bitwise alone-vs-batched check of the tuned table, dense grid")
a = ap.parse_args()

BUCKETS = (1, 4, 8, 16, 32, 64, 256, 512, 1024, 2048)  # same buckets as the existing Hopper entries
BLOCK_KS = (64, 128)
SPACE = list(itertools.product((16, 32, 64, 128), (32, 64, 128, 256), (4, 8), (2, 3, 4, 5)))
SMEM = 227 * 1024
if a.quick:
    BUCKETS, SPACE = (1, 32, 256), [(16, 64, 4, 3), (64, 128, 4, 3), (128, 128, 8, 3)]
dev = torch.device("cuda")
dt = torch.bfloat16  # vLLM uses the tuned table for bf16 only
orig_get = bi._get_matmul_config


def run_with(cfg):
    bi._get_matmul_config = lambda M, N, K, dtype, default: cfg


def bench(fn):
    return triton.testing.do_bench(fn, warmup=10, rep=40)


def tune_shape(n, k):
    g = torch.Generator(device=dev).manual_seed(0)
    w = (torch.randn(n, k, device=dev, dtype=dt, generator=g) * k ** -0.5).to(dt)
    x = torch.randn(max(BUCKETS), k, device=dev, dtype=dt, generator=g)
    res = {"cublas": {}, "default": {}, "by_bk": {}}
    bi._get_matmul_config = orig_get
    for m in BUCKETS:
        xm = x[:m]
        res["cublas"][m] = bench(lambda: F.linear(xm, w))
        res["default"][m] = bench(lambda: bi.linear_batch_invariant(xm, w))
    for bk in BLOCK_KS:
        per = {}
        for m in BUCKETS:
            xm = x[:m]
            best = None
            for bm, bn, nw, ns in SPACE:
                if ns * (bm * bk + bk * bn) * 2 > SMEM or (bm > 16 and bm > 4 * m):
                    continue
                cfg = {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "GROUP_SIZE_M": 8,
                       "num_warps": nw, "num_stages": ns}
                run_with(cfg)
                try:
                    t = bench(lambda: bi.linear_batch_invariant(xm, w))
                except Exception:  # out of resources etc.
                    continue
                if best is None or t < best[0]:
                    best = (t, [bm, bn, nw, ns])
            per[m] = best
        score = sum(math.log(per[m][0] / res["cublas"][m]) for m in BUCKETS) / len(BUCKETS)
        res["by_bk"][bk] = {"score": score, "per": per}
        print(f"  {n}x{k} BLOCK_K={bk} mean log(t/cublas)={score:.3f}", flush=True)
    bi._get_matmul_config = orig_get
    bk = min(BLOCK_KS, key=lambda b: res["by_bk"][b]["score"])
    per = res["by_bk"][bk]["per"]
    entry = {"n": n, "k": k, "block_k": bk, "m_buckets": [[m] + per[m][1] for m in BUCKETS],
             "ms": {m: {"tuned": per[m][0], "default": res["default"][m], "cublas": res["cublas"][m]} for m in BUCKETS},
             "score_by_block_k": {b: res["by_bk"][b]["score"] for b in BLOCK_KS}}
    for m in BUCKETS:
        r = entry["ms"][m]
        print(f"    M={m:<5} tuned {r['tuned']*1e3:8.1f}us  default {r['default']*1e3:8.1f}us  cublas {r['cublas']*1e3:8.1f}us",
              flush=True)
    return entry


def install(entries):
    table = bc._BATCH_INVARIANT_MATMUL_TUNED_CONFIGS.setdefault("hopper", {})
    for e in entries:
        table[(e["n"], e["k"])] = bc._MatmulShapeConfig(
            block_k=e["block_k"], m_buckets=tuple((b[0], bc._MatmulMConfig(*b[1:])) for b in e["m_buckets"]))
    bc._TUNED_MATMUL_CONFIGS_RESOLVED = False


def check(entries):
    """Rows alone vs in a batch, batch sizes 1..1024 then every 16th to 8192, first/last 8 and 4 middle rows."""
    install(entries)
    bi._get_matmul_config = orig_get
    ms = list(range(1, 1025)) + list(range(1040, 8193, 16))
    if a.quick:
        ms = list(range(1, 40)) + [255, 256, 257, 1024]
    out = {}
    for e in entries:
        n, k = e["n"], e["k"]
        g = torch.Generator(device=dev).manual_seed(1)
        w = (torch.randn(n, k, device=dev, dtype=dt, generator=g) * k ** -0.5).to(dt)
        x = torch.randn(max(ms), k, device=dev, dtype=dt, generator=g)
        alone, bad, rows, bad_m = {}, 0, 0, []
        for m in ms:
            idx = sorted(set(range(min(m, 8))) | set(range(max(0, m - 8), m))
                         | {i for i in (m // 5, 2 * m // 5, 3 * m // 5, 4 * m // 5) if i < m})
            for i in idx:
                if i not in alone:
                    alone[i] = bi.linear_batch_invariant(x[i:i + 1], w)
            o = bi.linear_batch_invariant(x[:m], w)
            b = sum(bool((o[i] != alone[i][0]).any()) for i in idx)
            bad += b
            rows += len(idx)
            if b:
                bad_m.append(m)
        out[f"{n}x{k}"] = {"bad": bad, "rows": rows, "bad_m": bad_m[:50]}
        print(f"  check {n}x{k}: {bad}/{rows} rows differ", flush=True)
        del x, w, alone
        torch.cuda.empty_cache()
    return out


shapes = [tuple(int(v) for v in s.split("x")) for s in a.shapes.split(";")]
print("torch", torch.__version__, "gpu", torch.cuda.get_device_name(), flush=True)
t0 = time.time()
entries = []
for n, k in shapes:
    entries.append(tune_shape(n, k))
    print(f"tuned {n}x{k} in {time.time() - t0:.0f}s", flush=True)
    json.dump({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "entries": entries}, open(a.out, "w"), indent=1)
result = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "entries": entries}
if a.check:
    result["check"] = check(entries)
    print("CHECK total bad", sum(v["bad"] for v in result["check"].values()), flush=True)
json.dump(result, open(a.out, "w"), indent=1)
