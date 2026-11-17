#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --export=ALL
# Bitwise alone-vs-batched check of the batch-invariant table in a vLLM source tree, through vLLM's own lookup.
# Fails (non-zero exit) if any row differs, so dependent jobs (afterok) do not start.
# Usage: sbatch --job-name=VL_<desc> --export=ALL,SRC_TREE=<tree>,SHAPES="NxK;...",OUT=<json>[,MAXM=16384] slurm/check_table.sbatch
set -eu
ROOT=${ROOT:-/home/it238/projects/vllm-batch-invariance}
VENV=${VENV:-/nobackup/autodelete/usr/it238/research_lab/vllm-pr-check/.venv}
export OPENSSL_CONF=/dev/null PYTHONPATH=$SRC_TREE
nvidia-smi --query-gpu=name,driver_version --format=csv
git -C "$SRC_TREE" rev-parse HEAD
"$VENV/bin/python" "$ROOT/src/tune_bi.py" --shapes "$SHAPES" --check-only --check-max-m "${MAXM:-16384}" --out "$OUT" | tee "${OUT%.json}.log"
grep -q "^CHECK total bad 0$" "${OUT%.json}.log"
