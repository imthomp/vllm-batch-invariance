#!/usr/bin/env python3
"""Decode seconds per token at each batch size.

Two output lengths at each batch size cancel prefill: step = dt / dtokens.
"""
import argparse
import json
import time

import numpy as np
from vllm import LLM, SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="Qwen/Qwen3-8B")
a = ap.parse_args()
llm = LLM(a.model, dtype="auto", gpu_memory_utilization=0.9, max_model_len=4096, seed=0)
SHORT, LONG = 64, 512


def run(n, k):
    ps = [f"Count upward from {i}: " for i in range(n)]
    t = time.time()
    llm.generate(ps, SamplingParams(temperature=0.6, max_tokens=k, ignore_eos=True),
                 use_tqdm=False)
    return time.time() - t


run(8, SHORT)  # warm caches
rows = []
for n in (1, 2, 4, 8, 16, 32, 64, 128):
    for rep in range(3):
        rows.append({"n": n, "rep": rep,
                     "step_s": (run(n, LONG) - run(n, SHORT)) / (LONG - SHORT)})
        print(rows[-1], flush=True)
ns = np.array([r["n"] for r in rows], float)
st = np.array([r["step_s"] for r in rows])
b, c = np.polyfit(ns, st, 1)
json.dump({"step_a": float(c), "step_b": float(b), "model": a.model, "rows": rows},
          open(a.out, "w"), indent=1)
print("step_a", c, "step_b", b)
