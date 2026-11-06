#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
# Phase 1f (PROTOCOL.md). Usage: sbatch --export=ALL,PY=...,OUT=...,CB=<extracted nvidia-cublas 13.8 wheel> slurm/cublas_bi_diag.sbatch
set -eu
S=${SRC:-$(pwd)/src}
SH="32x2880;97x2880;127x4096;129x2880;128x2880;4608x3584;152064x3584"
A=(--dense --trials 1 --catch --shapes "$SH")
N138="PYTHONPATH=$CB VBI_EXPECT_CUBLASLT=130802"
nvidia-smi --query-gpu=name,driver_version --format=csv
for arm in auto:8:lt,rp,fp32 align2:1:lt,rp,fp32 align16:4:lt,rp,fp32 auto_lt:8:lt auto_rp:8:rp; do
  IFS=: read name fl tf <<< "$arm"
  echo "== $name flags=$fl torch=$tf"
  CUBLAS_BATCH_INVARIANCE_FLAGS=$fl env $N138 "$PY" "$S/bitwise2.py" --mode bi_noenv --torch-flags "$tf" --out "$OUT/F_$name.json" "${A[@]}" 2>&1 | grep -E "SUMMARY|ERROR|Assert" | cut -c1-300
done
