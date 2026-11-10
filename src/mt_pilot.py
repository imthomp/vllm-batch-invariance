#!/usr/bin/env python3
"""MT pilot: translate WMT23 test sets several times, changing only serving state (order and batch grouping)."""
import argparse
import json
import random
from pathlib import Path

from vllm import LLM, SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--sacrebleu", default=str(Path.home() / ".sacrebleu/wmt23"))
a = ap.parse_args()

PAIRS = {"en-de": ("English", "German", 557), "zh-en": ("Chinese", "English", 400)}
RUNS = [("orig_all", None, 0), ("shuf_all", 1, 0), ("shuf_64", 2, 64), ("shuf_16", 3, 16), ("shuf_4", 4, 4)]

llm = LLM(a.model, dtype="bfloat16", seed=0, max_model_len=4096, gpu_memory_utilization=0.9)
tok = llm.get_tokenizer()
sp = SamplingParams(temperature=0.0, max_tokens=768)


def prompt(src, sl, tl):
    msg = f"Translate the following {sl} text into {tl}. Output only the translation.\n\n{src}"
    return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)


out = open(a.out, "w")
for pair, (sl, tl, n) in PAIRS.items():
    srcs = Path(f"{a.sacrebleu}/wmt23.{pair}.src").read_text().splitlines()[:n]
    ps = [prompt(s, sl, tl) for s in srcs]
    for name, seed, chunk in RUNS:
        order = list(range(n))
        if seed is not None:
            random.Random(seed).shuffle(order)
        size = chunk or n
        hyps = {}
        for start in range(0, n, size):
            idx = order[start:start + size]
            for i, o in zip(idx, llm.generate([ps[i] for i in idx], sp, use_tqdm=False)):
                hyps[i] = o.outputs[0].text.strip()
        for i in range(n):
            out.write(json.dumps({"pair": pair, "seg": i, "run": name, "hyp": hyps[i]}, ensure_ascii=False) + "\n")
        out.flush()
        print(pair, name, "done", flush=True)
print("DONE", flush=True)
