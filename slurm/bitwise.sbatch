#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
# Phase 1. Usage: sbatch --export=ALL,PY=<python with torch>,TAG=<name>,OUT=<dir>[,STRESS=1] slurm/bitwise.sbatch
set -eu
S=${SRC:-$(pwd)/src}
nvidia-smi --query-gpu=name,driver_version --format=csv
for mode in bi bi_noenv default; do
  "$PY" "$S/bitwise2.py" --mode $mode --out "$OUT/${TAG}_$mode.json" ${STRESS:+--stress}
done
