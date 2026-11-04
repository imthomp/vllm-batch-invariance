#!/usr/bin/env python3
"""Minimal repro: Gemma-2-9B (sliding-window attention; max_model_len must exceed its 4096 window) is not batch-invariant under VLLM_BATCH_INVARIANT=1 with FLASH_ATTN and CUDA graphs.

Each prompt is generated greedily alone and inside a batch of unrelated prompts; with batch invariance the two
outputs must match token for token. Usage:
  VLLM_BATCH_INVARIANT=1 python repro_gemma2.py                         # expected: some prompts differ
  VLLM_BATCH_INVARIANT=1 python repro_gemma2.py --backend TRITON_ATTN   # expected: none differ
  VLLM_BATCH_INVARIANT=1 python repro_gemma2.py --no-cudagraph          # expected: none differ
"""
import argparse
import random

from vllm import LLM, SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="google/gemma-2-9b-it")
ap.add_argument("--n", type=int, default=20, help="prompts to test")
ap.add_argument("--batch", type=int, default=16, help="batch size for the batched run")
ap.add_argument("--backend", default="")
ap.add_argument("--no-cudagraph", action="store_true")
a = ap.parse_args()

kw = dict(dtype="bfloat16", seed=0, max_model_len=8192, enable_prefix_caching=False)
if a.backend:
    kw["attention_config"] = {"backend": a.backend}
if a.no_cudagraph:
    kw["compilation_config"] = {"cudagraph_mode": "NONE"}
llm = LLM(a.model, **kw)

rng = random.Random(0)
words = "the river stone light market winter number garden letter engine silver morning question".split()
background = " ".join(f"Note {i}: the {rng.choice(words)} near the {rng.choice(words)} was {rng.randint(1, 999)}."
                      for i in range(220))
# Prompts of varied length, so batched requests are at different positions.
notes = background.split(". ")
prompts = [". ".join(notes[: 60 + (37 * i) % 160]) + f".\n\nUsing the notes above, add the values of notes {i % 50}, "
           f"{i % 50 + 7} and {i % 50 + 19}, then explain each step." for i in range(a.n + a.batch)]
sp = SamplingParams(temperature=0.0, max_tokens=256)

alone = [llm.generate([p], sp, use_tqdm=False)[0].outputs[0].token_ids for p in prompts[:a.n]]
differ = 0
for i in range(a.n):
    batch = [prompts[i]] + rng.sample(prompts[a.n:], a.batch - 1)
    together = llm.generate(batch, sp, use_tqdm=False)[0].outputs[0].token_ids
    if list(together) != list(alone[i]):
        differ += 1
        first = next((j for j, (x, y) in enumerate(zip(together, alone[i])) if x != y), min(len(together), len(alone[i])))
        print(f"prompt {i}: alone and batched outputs differ from token {first}")
print(f"RESULT {differ}/{a.n} prompts differ between alone and batched")
