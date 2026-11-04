#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3:00:00
# Tuning pilot (PROTOCOL.md). Usage: sbatch --export=ALL,PY=<python with vllm>,OUT=<dir>,SHAPES="NxK;..." slurm/tune.sbatch
set -eu
S=${SRC:-$(pwd)/src}
nvidia-smi --query-gpu=name,driver_version --format=csv
"$PY" "$S/tune_bi.py" --shapes "$SHAPES" --out "$OUT/tuned_${TAG:-q25}.json" --check
