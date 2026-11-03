#!/usr/bin/env python3
"""Phase 2 crossover: same requests under cold/warm prefix cache, alone or with 7 unrelated neighbors.

Adds tensor parallelism, long few-shot prompts, a small chunked-prefill budget, and seeded sampling to crossover.py.
Each request runs 8 times (4 conditions, twice each, random order); every pair of runs is compared.
"""
import argparse
import json
import random
import time

from vllm import LLM, SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--max-new", type=int, default=256)
ap.add_argument("--tp", type=int, default=1)
ap.add_argument("--shots", type=int, default=40, help="few-shot examples in the shared prefix")
ap.add_argument("--max-len", type=int, default=8192)
ap.add_argument("--chunk", type=int, default=0, help="max_num_batched_tokens; 0 keeps vLLM's default")
ap.add_argument("--temp", type=float, default=0.0, help=">0 uses seeded sampling, seed = request index")
a = ap.parse_args()
probs = json.load(open(a.data))
random.Random(0).shuffle(probs)
gsm = [p for p in probs if p["src"] == "gsm8k"]
shots = gsm[:a.shots]
rest = [p for p in probs if p not in shots]
test, filler = rest[:a.n], rest[a.n:a.n + 400]
pre = "Solve each problem. Finish with the answer in \\boxed{}.\n\n" + "".join(
    f"Problem: {p['q']}\nAnswer: \\boxed{{{p['a']}}}\n\n" for p in shots)
kw = dict(dtype="bfloat16", gpu_memory_utilization=0.9, max_model_len=a.max_len, seed=0,
          enable_prefix_caching=True, tensor_parallel_size=a.tp)
if a.chunk:
    kw.update(enable_chunked_prefill=True, max_num_batched_tokens=a.chunk)
llm = LLM(a.model, **kw)
tok = llm.get_tokenizer()
tkw = {"enable_thinking": False} if "qwen3" in a.model.lower() else {}


def enc(p):
    msgs = [{"role": "user", "content": f"{pre}Problem: {p['q']}\nAnswer:"}]
    return tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **tkw),
               add_special_tokens=False).input_ids


ids, fill = [enc(p) for p in test], [enc(p) for p in filler]
print("prompt tokens: median", sorted(map(len, ids))[len(ids) // 2], "max", max(map(len, ids)), flush=True)


def sp(i):
    if a.temp > 0:
        return SamplingParams(temperature=a.temp, top_p=1.0, max_tokens=a.max_new, logprobs=20, seed=i)
    return SamplingParams(temperature=0.0, max_tokens=a.max_new, logprobs=20)


out = open(a.out, "w")
rng = random.Random(1)
t0 = time.time()
for i, x in enumerate(ids):
    conds = ["cold_alone", "warm_alone", "cold_mixed", "warm_mixed"] * 2
    rng.shuffle(conds)
    llm.generate([{"prompt_token_ids": x, "cache_salt": f"p{i}"}], sp(i), use_tqdm=False)
    for r, c in enumerate(conds):
        salt = f"p{i}" if c.startswith("warm") else f"c{i}-{r}"
        ps = [{"prompt_token_ids": x, "cache_salt": salt}]
        params = [sp(i)]
        if c.endswith("mixed"):
            ps += [{"prompt_token_ids": f, "cache_salt": f"f{rng.random()}"} for f in rng.sample(fill, 7)]
            params += [sp(10_000 + j) for j in range(7)]
        o = llm.generate(ps, params, use_tqdm=False)[0]
        k = o.outputs[0]
        out.write(json.dumps({"i": i, "cond": c, "rep": r, "t": list(k.token_ids),
                              "top": [{str(t): v.logprob for t, v in d.items()} for d in k.logprobs][:64],
                              "finish": k.finish_reason, "plen": len(x),
                              "hit": (o.num_cached_tokens or 0) / len(x)}) + "\n")
    out.flush()
print("DONE requests", len(ids), f"{time.time() - t0:.0f}s", flush=True)
