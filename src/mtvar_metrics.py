#!/usr/bin/env python3
"""Score MT eval-variance translations with MetricX-24, COMET-22 and CometKiwi-23 XL (PROTOCOL.md MT study).

Every unique (segment id, hypothesis) pair across all translate runs is scored exactly once, in a fixed order at batch
size 32, so two runs with the same translation get the same score by construction. COMET models skip zu_ZA (not covered
by XLM-R). Runs in the multilingual-mqm-benchmark venv (comet + its MetricX implementation).
Also the side measure: COMET-22 and MetricX of one fixed run (--side-run) at batch sizes 1, 16 and 64.
Usage: mtvar_metrics.py --root <translate dir> --out <scores.jsonl.gz> [--side-run <run.jsonl> --side-out <json>]
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

MQM_REPO = Path("/home/it238/projects/multilingual-mqm-benchmark/multilingual-mqm-benchmark/src")
COMET22 = "Unbabel/wmt22-comet-da"
KIWI = "Unbabel/wmt23-cometkiwi-da-xl"
NO_COMET = {"zu_ZA"}
BS = 32


def load(p):
    p = Path(p)
    return [json.loads(x) for x in (gzip.open(p, "rt") if p.suffix == ".gz" else p.open())]


def comet_scores(name, triples, bs, ref=True):
    from comet import download_model, load_from_checkpoint
    model = load_from_checkpoint(download_model(name))
    data = [{"src": s, "mt": h, **({"ref": r} if ref else {})} for s, h, r in triples]
    return list(model.predict(data, batch_size=bs, gpus=1, progress_bar=False).scores)


def metricx_scores(triples, bs):
    sys.path.insert(0, str(MQM_REPO))
    from mqmbench.metrics import metricx
    return metricx.score([t[0] for t in triples], [t[1] for t in triples], [t[2] for t in triples], batch_size=bs)


ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--side-run")
ap.add_argument("--side-out")
ap.add_argument("--bs", type=int, default=BS, help="scoring batch size (J5 extension scores all translations at 1 and 64)")
a = ap.parse_args()

uniq = {}
for f in sorted(Path(a.root).glob("*/bi*/C*_r*.jsonl*")):
    if f.name.endswith(".meta.json"):
        continue
    for r in load(f):
        uniq.setdefault((r["id"], r["text"].strip()), (r["lp"], r["src"], r["ref"]))
keys = sorted(uniq)
print(len(keys), "unique (segment, hypothesis) pairs", flush=True)
trip = [(uniq[k][1], k[1], uniq[k][2]) for k in keys]
mx = metricx_scores(trip, a.bs)
cidx = [i for i, k in enumerate(keys) if uniq[k][0] not in NO_COMET]
c22 = comet_scores(COMET22, [trip[i] for i in cidx], a.bs)
kiwi = comet_scores(KIWI, [trip[i] for i in cidx], a.bs, ref=False)
cmap = {i: (c22[j], kiwi[j]) for j, i in enumerate(cidx)}
with gzip.open(a.out, "wt") as g:
    for i, k in enumerate(keys):
        c = cmap.get(i, (None, None))
        g.write(json.dumps({"id": k[0], "hyp": k[1], "metricx": mx[i], "comet22": c[0], "kiwi": c[1]},
                           ensure_ascii=False) + "\n")
print("wrote", a.out, flush=True)

if a.side_run:
    recs = load(a.side_run)
    t = [(r["src"], r["text"].strip(), r["ref"]) for r in recs]
    tc = [x for x, r in zip(t, recs) if r["lp"] not in NO_COMET]
    side = {"run": a.side_run, "n_metricx": len(t), "n_comet": len(tc)}
    for m, fn, data in (("metricx", metricx_scores, t), ("comet22", lambda d, b: comet_scores(COMET22, d, b), tc)):
        by = {b: fn(data, b) for b in (1, 16, 64)}
        for b in (16, 64):
            diff = [abs(x - y) for x, y in zip(by[1], by[b])]
            side[f"{m}.bs1_vs_bs{b}.differ"] = sum(d != 0 for d in diff)
            side[f"{m}.bs1_vs_bs{b}.max_abs"] = max(diff)
    Path(a.side_out).write_text(json.dumps(side, indent=1))
    print(json.dumps(side))
