#!/usr/bin/env python3
"""Does vLLM's batch-invariant mode start and generate for a model? Real architecture, few layers, dummy weights.
Usage: VLLM_BATCH_INVARIANT=1 python src/bi_smoke.py <model path or id> [--layers 4] [--tp 1]"""
import argparse
import os

from vllm import LLM, SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--tp", type=int, default=1)
a = ap.parse_args()


def shrink(cfg):
    for c in (cfg, getattr(cfg, "text_config", None)):
        if c is not None and hasattr(c, "num_hidden_layers"):
            c.num_hidden_layers = a.layers
    return cfg


print("VLLM_BATCH_INVARIANT", os.environ.get("VLLM_BATCH_INVARIANT"), "model", a.model, flush=True)
llm = LLM(a.model, load_format="dummy", hf_overrides=shrink, tensor_parallel_size=a.tp, max_model_len=2048,
          gpu_memory_utilization=0.85, trust_remote_code=True, seed=0)
out = llm.generate(["The capital of France is"] * 4, SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True))
print("SMOKE OK", a.model, [len(o.outputs[0].token_ids) for o in out], flush=True)
