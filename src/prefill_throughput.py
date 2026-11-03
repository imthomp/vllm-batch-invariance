#!/usr/bin/env python3
"""Prefill seconds per prompt token at several batch sizes and prompt lengths, prefix caching off."""
import argparse
import json
import random
import time

from vllm import LLM, SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--tp", type=int, default=1)
a = ap.parse_args()
llm = LLM(a.model, dtype="bfloat16", gpu_memory_utilization=0.9, max_model_len=8192, seed=0,
          enable_prefix_caching=False, tensor_parallel_size=a.tp)
vocab = llm.get_tokenizer().vocab_size
rng = random.Random(0)
sp = SamplingParams(temperature=0.0, max_tokens=1)


def run(n, length):
    ps = [{"prompt_token_ids": [rng.randrange(1000, vocab - 1000) for _ in range(length)]} for _ in range(n)]
    t = time.time()
    llm.generate(ps, sp, use_tqdm=False)
    return time.time() - t


run(4, 1024)  # warm up
rows = []
for length in (1024, 4096):
    for n in (1, 8, 32):
        for rep in range(3):
            dt = run(n, length)
            rows.append({"len": length, "n": n, "rep": rep, "s_per_tok": dt / (n * length)})
            print(rows[-1], flush=True)
json.dump({"model": a.model, "tp": a.tp, "rows": rows}, open(a.out, "w"), indent=1)
