#!/usr/bin/env python3
"""Eval-variance generation run (PROTOCOL.md "Do serving conditions change NLP evaluation conclusions?").

One process = one (model, condition, repeat, batch-invariant mode). VLLM_BATCH_INVARIANT is read from the environment.
Writes one JSONL record per benchmark prompt (filler outputs in C3 are discarded) plus a .meta.json with timing.
Usage: evalvar_gen.py --model <hf id> --cond C1|C2|C3|C4 --out <path.jsonl> [--limit N]
"""
import argparse
import json
import os
import time
from pathlib import Path

DATA = Path("/nobackup/autodelete/usr/it238/research_lab/data")
INSTR = "\n\nPut your final answer within \\boxed{}."
CONDS = {  # engine settings per condition; C3 adds filler prompts
    "C1": dict(max_num_seqs=1),
    "C2": dict(max_num_seqs=256),
    "C3": dict(max_num_seqs=256),
    "C4": dict(max_num_seqs=256, max_num_batched_tokens=512, enable_chunked_prefill=True),
}

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--cond", required=True, choices=sorted(CONDS))
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0, help="smoke only: first N benchmark prompts")
a = ap.parse_args()

from vllm import LLM, SamplingParams  # noqa: E402
import torch  # noqa: E402
import vllm  # noqa: E402

probs = json.loads((DATA / "problems.json").read_text())
assert len(probs) == 1819 and sum(p["src"] == "gsm8k" for p in probs) == 1319
if a.limit:
    probs = probs[: a.limit // 2] + [p for p in probs if p["src"] == "math500"][: a.limit - a.limit // 2]
convs = [[{"role": "user", "content": p["q"] + INSTR}] for p in probs]
is_bench = [True] * len(convs)
if a.cond == "C3":
    hot = json.loads((DATA / "hotpot.json").read_text())[: len(probs)]
    filler = [[{"role": "user", "content": f"{h['paras'][0]['title']}: {h['paras'][0]['text']}\n\nQuestion: {h['q']}"}]
              for h in hot]
    convs = [c for pair in zip(convs, filler) for c in pair]
    is_bench = [b for _ in probs for b in (True, False)]

llm = LLM(model=a.model, dtype="bfloat16", seed=0, max_model_len=4096, gpu_memory_utilization=0.9,
          enable_prefix_caching=True, **CONDS[a.cond])
sp = SamplingParams(temperature=0.0, max_tokens=1024, seed=0)
t0 = time.time()
outs = llm.chat(convs, sp, use_tqdm=False)
gen_s = time.time() - t0

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
n_tok = 0
with open(a.out, "w") as f:
    bench = [o for o, b in zip(outs, is_bench) if b]
    assert len(bench) == len(probs)
    for p, o in zip(probs, bench):
        c = o.outputs[0]
        n_tok += len(c.token_ids)
        f.write(json.dumps({"id": p["id"], "src": p["src"], "gold": p["a"], "text": c.text,
                            "finish": c.finish_reason, "token_ids": list(c.token_ids)}) + "\n")
meta = {"model": a.model, "cond": a.cond, "bi": os.environ.get("VLLM_BATCH_INVARIANT", "0"), "engine": CONDS[a.cond],
        "n_bench": len(probs), "n_total": len(convs), "gen_seconds": gen_s, "bench_output_tokens": n_tok,
        "gpu": torch.cuda.get_device_name(0), "vllm": vllm.__version__, "vllm_src": os.environ.get("PYTHONPATH", ""),
        "job": os.environ.get("SLURM_JOB_ID", "local"), "limit": a.limit}
Path(a.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
print("DONE", json.dumps(meta))
