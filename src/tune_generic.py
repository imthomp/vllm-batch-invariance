#!/usr/bin/env python3
"""Choose one generic default configuration for vLLM's batch-invariant matmul (PROTOCOL.md "Better default").

grid:   time every candidate configuration at each bucket's largest batch size for every training shape, BLOCK_K 64 and 128.
select: per bucket, the candidate with the highest geometric-mean speedup over vLLM's current default across training
        shapes (candidates that fail on any shape are excluded); the BLOCK_K kept has the higher overall geometric mean.
check:  bitwise alone-vs-batched check of the selected default on held-out shapes (batch sizes 1-1024, every 16th to 8192).
Usage: tune_generic.py --train "NxK;..." --test "NxK;..." --out generic.json [--quick]
"""
import argparse
import itertools
import json
import math
import time

import torch
import triton.testing

import vllm.model_executor.determinism.batch_invariant as bi

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True)
ap.add_argument("--test", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--quick", action="store_true", help="tiny grid, smoke tests only")
a = ap.parse_args()

BUCKETS = (1, 4, 8, 16, 32, 64, 256, 512, 1024, 2048)
BLOCK_KS = (64, 128)
SPACE = list(itertools.product((16, 32, 64, 128), (32, 64, 128, 256), (4, 8), (2, 3, 4, 5)))
if a.quick:
    BUCKETS, SPACE = (1, 32, 256), [(16, 64, 4, 3), (64, 128, 4, 3), (128, 128, 8, 3)]
SMEM = 227 * 1024
dev, dt = torch.device("cuda"), torch.bfloat16
orig_get = bi._get_matmul_config
parse = lambda s: [tuple(int(v) for v in x.split("x")) for x in s.split(";")]
train, test = parse(a.train), parse(a.test)
assert not set(train) & set(test), "test shapes must be held out"


def cfg(bm, bn, bk, nw, ns):
    return {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "GROUP_SIZE_M": 8, "num_warps": nw, "num_stages": ns}


def bench(f):
    return triton.testing.do_bench(f, warmup=5, rep=25)


def candidates(m, bk):
    return [c for c in SPACE if c[3] * (c[0] * bk + bk * c[1]) * 2 <= SMEM and not (c[0] > 16 and c[0] > 4 * m)]


print("torch", torch.__version__, "gpu", torch.cuda.get_device_name(), flush=True)
t0 = time.time()
grid = {}  # (shape, bk, m) -> {cand: ms}; plus default times
default = {}
for n, k in train:
    g = torch.Generator(device=dev).manual_seed(0)
    w = (torch.randn(n, k, device=dev, dtype=dt, generator=g) * k ** -0.5).to(dt)
    x = torch.randn(max(BUCKETS), k, device=dev, dtype=dt, generator=g)
    for m in BUCKETS:
        xm = x[:m]
        bi._get_matmul_config = orig_get
        default[(n, k, m)] = bench(lambda: bi.linear_batch_invariant(xm, w))
        for bk in BLOCK_KS:
            res = {}
            for c in candidates(m, bk):
                bi._get_matmul_config = lambda M, N, K, dtype, d, c=c: cfg(c[0], c[1], bk, c[2], c[3])
                try:
                    res[c] = bench(lambda: bi.linear_batch_invariant(xm, w))
                except Exception:
                    pass
            grid[(n, k, bk, m)] = res
    bi._get_matmul_config = orig_get
    del w, x
    torch.cuda.empty_cache()
    print(f"grid {n}x{k} done, {time.time() - t0:.0f}s", flush=True)

selected, score = {}, {}
for bk in BLOCK_KS:
    buckets, logs = [], []
    for m in BUCKETS:
        common = set.intersection(*[set(grid[(n, k, bk, m)]) for n, k in train])
        best = max(common, key=lambda c: sum(math.log(default[(n, k, m)] / grid[(n, k, bk, m)][c]) for n, k in train))
        gm = math.exp(sum(math.log(default[(n, k, m)] / grid[(n, k, bk, m)][best]) for n, k in train) / len(train))
        buckets.append([m, *best])
        logs.append(math.log(gm))
        print(f"  BLOCK_K={bk} M<={m:<5} {best}  geomean speedup over default {gm:.2f}x", flush=True)
    selected[bk] = buckets
    score[bk] = math.exp(sum(logs) / len(logs))
    print(f"BLOCK_K={bk}: overall geomean speedup {score[bk]:.3f}x", flush=True)
bk = max(BLOCK_KS, key=lambda b: score[b])
generic = {"block_k": bk, "m_buckets": selected[bk]}
print("SELECTED", generic, flush=True)


def generic_cfg(M):
    for row in generic["m_buckets"]:
        if M <= row[0]:
            return cfg(row[1], row[2], bk, row[3], row[4])
    r = generic["m_buckets"][-1]
    return cfg(r[1], r[2], bk, r[3], r[4])


# Held-out: bitwise check and speed against the current default.
bi._get_matmul_config = lambda M, N, K, dtype, d: generic_cfg(M)
ms = list(range(1, 1025)) + list(range(1040, 8193, 16))
if a.quick:
    ms = list(range(1, 40)) + [255, 256, 257, 1024]
check, speed = {}, {}
for n, k in test:
    g = torch.Generator(device=dev).manual_seed(1)
    w = (torch.randn(n, k, device=dev, dtype=dt, generator=g) * k ** -0.5).to(dt)
    x = torch.randn(max(ms), k, device=dev, dtype=dt, generator=g)
    alone, bad, rows = {}, 0, 0
    for m in ms:
        idx = sorted(set(range(min(m, 8))) | set(range(max(0, m - 8), m)) | {i for i in (m // 5, 2 * m // 5, 3 * m // 5, 4 * m // 5) if i < m})
        for i in idx:
            if i not in alone:
                alone[i] = bi.linear_batch_invariant(x[i:i + 1], w)
        o = bi.linear_batch_invariant(x[:m], w)
        bad += sum(bool((o[i] != alone[i][0]).any()) for i in idx)
        rows += len(idx)
    check[f"{n}x{k}"] = {"bad": bad, "rows": rows}
    sp = {}
    for m in BUCKETS:
        xm = x[:m]
        bi._get_matmul_config = lambda M, N, K, dtype, d: generic_cfg(M)
        tg = bench(lambda: bi.linear_batch_invariant(xm, w))
        bi._get_matmul_config = orig_get
        td = bench(lambda: bi.linear_batch_invariant(xm, w))
        tc = bench(lambda: torch.nn.functional.linear(xm, w))
        sp[m] = {"generic": tg, "default": td, "cublas": tc}
    bi._get_matmul_config = lambda M, N, K, dtype, d: generic_cfg(M)
    speed[f"{n}x{k}"] = sp
    print(f"  test {n}x{k}: {bad}/{rows} rows differ; speedup over default " +
          " ".join(f"M{m}:{sp[m]['default'] / sp[m]['generic']:.2f}x" for m in BUCKETS), flush=True)
    del w, x, alone
    torch.cuda.empty_cache()
print("CHECK total bad", sum(v["bad"] for v in check.values()), flush=True)
json.dump({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "train": train, "test": test,
           "generic": generic, "score_by_block_k": score, "selected_by_block_k": selected,
           "check": check, "test_speed": speed,
           "default_ms": {f"{n}x{k}|{m}": v for (n, k, m), v in default.items()}}, open(a.out, "w"), indent=1)
