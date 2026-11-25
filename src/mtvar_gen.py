#!/usr/bin/env python3
"""MT eval-variance run (PROTOCOL.md "Do serving conditions change MT evaluation conclusions?").

Two tasks share the same serving conditions:
  --task translate: one system translates WMT24++ (7 directions, good sources only).
  --task judge:     the LLM judge annotates fixed translations (--items, from src/mtvar_items.py).
One process = one (model, task, condition, repeat, batch-invariant mode); VLLM_BATCH_INVARIANT is read from the env.
Writes one JSONL record per item plus a .meta.json with timing.
Usage: mtvar_gen.py --task translate|judge --model <hf id> --cond C1..C4 --out <path.jsonl> [--items f] [--limit N]
"""
import argparse
import json
import os
import time
from pathlib import Path

DATA = Path("/nobackup/autodelete/usr/it238/research_lab/data")
LANGS = {"de_DE": "German", "zh_CN": "Simplified Chinese", "ja_JP": "Japanese", "is_IS": "Icelandic",
         "sw_KE": "Swahili", "gu_IN": "Gujarati", "zu_ZA": "Zulu"}
CONDS = {
    "C1": dict(max_num_seqs=1),
    "C2": dict(max_num_seqs=256),
    "C3": dict(max_num_seqs=256),
    "C4": dict(max_num_seqs=256, max_num_batched_tokens=512, enable_chunked_prefill=True),
}
JUDGE = """You are an expert annotator of translation quality (MQM). Identify the errors in the {tl} translation of the English source below. For each error, write one line in the form
<severity>: <category> - <short explanation>
where severity is critical, major or minor and category is one of accuracy, fluency, terminology, style, locale convention, other. If the translation has no errors, write exactly: no-error

English source:
{src}

{tl} translation:
{hyp}

Errors:"""


def wmt24pp(limit):
    items = []
    for lp in LANGS:  # google/wmt24pp files, copied to DATA/wmt24pp on a login node
        recs = [json.loads(x) for x in open(DATA / "wmt24pp" / f"en-{lp}.jsonl")]
        good = [r for r in recs if not r["is_bad_source"]]
        assert len(good) == 960, (lp, len(good))
        for r in good[:limit] if limit else good:
            items.append({"id": f"{lp}:{r['segment_id']}", "lp": lp, "src": r["source"], "ref": r["target"]})
    return items


ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True, choices=["translate", "judge"])
ap.add_argument("--model", required=True)
ap.add_argument("--cond", required=True, choices=sorted(CONDS))
ap.add_argument("--out", required=True)
ap.add_argument("--items", help="judge items JSONL (src/mtvar_items.py)")
ap.add_argument("--limit", type=int, default=0, help="pilot only: first N segments per direction / N judge items")
a = ap.parse_args()

if a.task == "translate":
    items = wmt24pp(a.limit)
    msgs = [f"Translate the following English text into {LANGS[x['lp']]}. Output only the translation.\n\n{x['src']}"
            for x in items]
    max_tokens = 1024
else:
    items = [json.loads(x) for x in open(a.items)]
    items = items[: a.limit] if a.limit else items
    msgs = [JUDGE.format(tl=LANGS[x["lp"]], src=x["src"], hyp=x["hyp"]) for x in items]
    max_tokens = 512

from vllm import LLM, SamplingParams  # noqa: E402
import torch  # noqa: E402
import vllm  # noqa: E402

convs = [[{"role": "user", "content": m}] for m in msgs]
is_item = [True] * len(convs)
if a.cond == "C3":
    hot = json.loads((DATA / "hotpot.json").read_text())[: len(items)]
    filler = [[{"role": "user", "content": f"{h['paras'][0]['title']}: {h['paras'][0]['text']}\n\nQuestion: {h['q']}"}]
              for h in hot]
    convs = [c for pair in zip(convs, filler) for c in pair]
    is_item = [b for _ in items for b in (True, False)]

extra = {"limit_mm_per_prompt": {"image": 0}} if "gemma-3" in a.model.lower() else {}
llm = LLM(model=a.model, dtype="bfloat16", seed=0, max_model_len=4096, gpu_memory_utilization=0.9,
          enable_prefix_caching=True, **CONDS[a.cond], **extra)
sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=0)
t0 = time.time()
outs = llm.chat(convs, sp, use_tqdm=False)
gen_s = time.time() - t0

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
kept = [o for o, b in zip(outs, is_item) if b]
assert len(kept) == len(items)
n_tok = 0
with open(a.out, "w") as f:
    for x, o in zip(items, kept):
        c = o.outputs[0]
        n_tok += len(c.token_ids)
        f.write(json.dumps({**x, "text": c.text, "finish": c.finish_reason, "token_ids": list(c.token_ids)},
                           ensure_ascii=False) + "\n")
meta = {"task": a.task, "model": a.model, "cond": a.cond, "bi": os.environ.get("VLLM_BATCH_INVARIANT", "0"),
        "engine": CONDS[a.cond], "n_items": len(items), "n_total": len(convs), "gen_seconds": gen_s,
        "output_tokens": n_tok, "gpu": torch.cuda.get_device_name(0), "vllm": vllm.__version__,
        "vllm_src": os.environ.get("PYTHONPATH", ""), "job": os.environ.get("SLURM_JOB_ID", "local"), "limit": a.limit}
Path(a.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
print("DONE", json.dumps(meta))
