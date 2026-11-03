#!/usr/bin/env python3
"""Crossover: same requests under cold/warm prefix cache, alone or with 7 unrelated neighbors, greedy, top-20 logprobs.
"""
import argparse
import json
import random
from vllm import LLM, SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--n", type=int, default=150)
ap.add_argument("--max-new", type=int, default=256)
ap.add_argument("--conds", default="cold_alone,warm_alone,cold_mixed,warm_mixed")
a = ap.parse_args()
probs = json.load(open(a.data))
random.Random(0).shuffle(probs)
shots = [p for p in probs if p["src"] == "gsm8k"][:40]
rest = [p for p in probs if p not in shots]
test, filler = rest[:a.n], rest[a.n:a.n + 400]
pre = "Solve each problem. Finish with the answer in \\boxed{}.\n\n" + "".join(
    f"Problem: {p['q']}\nAnswer: \\boxed{{{p['a']}}}\n\n" for p in shots)
llm = LLM(a.model, dtype=a.dtype, gpu_memory_utilization=0.9, max_model_len=8192, seed=0, enable_prefix_caching=True)
tok = llm.get_tokenizer()
kw = {"enable_thinking": False} if "qwen" in a.model.lower() else {}
enc = lambda p: tok(tok.apply_chat_template([{"role": "user", "content": f"{pre}Problem: {p['q']}\nAnswer:"}],
                                            tokenize=False, add_generation_prompt=True, **kw), add_special_tokens=False).input_ids
ids, fill = [enc(p) for p in test], [enc(p) for p in filler]
sp = SamplingParams(temperature=0.0, max_tokens=a.max_new, logprobs=20)
out = open(a.out, "w")
rng = random.Random(1)


def gen(ps):
    return llm.generate(ps, sp, use_tqdm=False)


import time
t0 = time.time()
gen_tokens = 0
for i, x in enumerate(ids):
    conds = a.conds.split(",") * 2
    rng.shuffle(conds)
    gen([{"prompt_token_ids": x, "cache_salt": f"p{i}"}])
    for r, c in enumerate(conds):
        salt = f"p{i}" if c.startswith("warm") else f"c{i}-{r}"
        ps = [{"prompt_token_ids": x, "cache_salt": salt}]
        if c.endswith("mixed"):
            ps += [{"prompt_token_ids": f, "cache_salt": f"f{rng.random()}"} for f in rng.sample(fill, 7)]
        o = gen(ps)[0]
        k = o.outputs[0]
        gen_tokens += len(k.token_ids)
        out.write(json.dumps({"i": i, "gold": test[i]["a"], "cond": c, "rep": r, "t": list(k.token_ids),
                              "top": [{str(t): v.logprob for t, v in d.items()} for d in k.logprobs][:64],
                              "text": k.text[-400:], "finish": k.finish_reason,
                              "hit": (o.num_cached_tokens or 0) / len(x)}) + "\n")
    out.flush()
print("THROUGHPUT single-target tok/s", gen_tokens / (time.time() - t0), "requests", len(ids), flush=True)
