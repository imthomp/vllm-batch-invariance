#!/bin/bash
# Cluster rules for agents: see BYU_ORC_AGENTS.md at the repo root (BYU ORC policy for AI agents).
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --export=ALL
# Archive one judge-robustness configuration and score it (PROTOCOL.md extension, A1-A4).
# Usage: sbatch --job-name=VL_mtx_score_<tag> --export=ALL,TAG=<j1|j2|j3>,SRC_DIR=<raw config dir>[,ITEMS=<file>][,REFS='h200=dir tp2=dir'] slurm/mtvar_ext_score.sbatch
set -euo pipefail
ROOT=/home/it238/projects/vllm-batch-invariance
PY=/nobackup/autodelete/usr/it238/research_lab/vllm-pr-check/.venv/bin/python
export OPENSSL_CONF=/dev/null
cd "$ROOT"
TAG=${TAG:?}; SRC_DIR=${SRC_DIR:?}
DEST=ext_$TAG
"$PY" src/mtvar_archive.py --src "$SRC_DIR" --dest "$DEST" ${ITEMS:+--items "$ITEMS"}
JD=$(ls -d results/validation/mtvar/$DEST/judge/*)
"$PY" - "$JD" <<'PY'
import gzip, json, sys
from pathlib import Path
d = Path(sys.argv[1]) / "bi1"
runs = sorted(d.glob("C*_r*.jsonl.gz"))
sha = {f.name: {r["id"]: r["token_sha"] for r in map(json.loads, gzip.open(f, "rt"))} for f in runs}
base = sha[runs[0].name]
for k, v in sha.items():
    nd = sum(v[i] != base[i] for i in base)
    print(f"BI-on {k}: {nd} of {len(base)} items differ from {runs[0].name}")
PY
refs=()
for r in ${REFS:-}; do refs+=(--ref "$r"); done
"$PY" src/mtvar_ext_score.py --tag "$TAG" --judge-dir "$JD" "${refs[@]}" --merge
