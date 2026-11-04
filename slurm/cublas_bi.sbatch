#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3:00:00
# Phase 1e (PROTOCOL.md). Usage: sbatch --export=ALL,PY=<python with torch>,OUT=<dir>,LT=<dir with cuBLAS 13.8 libs> slurm/cublas_bi.sbatch
set -eu
S=${SRC:-$(pwd)/src}
SH="32x2880;64x2880;96x2880;97x2880;100x2880;104x2880;127x2880;128x2880;129x2880;256x2880;64x4096;96x4096;127x4096;128x4096;129x4096;256x4096;64x5120;96x5120;128x5120;64x2048;96x2048;4608x3584;3584x18944;28672x4096;152064x3584"
A=(--dense --trials 1 --catch --shapes "$SH")
P138="$LT/libcublasLt.so.13:$LT/libcublas.so.13"
nvidia-smi --query-gpu=name,driver_version --format=csv
echo "== A"; "$PY" "$S/bitwise2.py" --mode bi --out "$OUT/A_lib131_bi.json" "${A[@]}"
echo "== B"; LD_PRELOAD=$P138 "$PY" "$S/bitwise2.py" --mode bi_noenv --out "$OUT/B_lib138_bi_noenv.json" "${A[@]}"
echo "== C"; CUBLAS_BATCH_INVARIANCE_FLAGS=8 LD_PRELOAD=$P138 "$PY" "$S/bitwise2.py" --mode bi_noenv --out "$OUT/C_lib138_flags8_bi_noenv.json" "${A[@]}"
echo "== D"; CUBLAS_BATCH_INVARIANCE_FLAGS=8 LD_PRELOAD=$P138 "$PY" "$S/bitwise2.py" --mode default --out "$OUT/D_lib138_flags8_default.json" "${A[@]}"
echo "== E"; CUBLAS_BATCH_INVARIANCE_FLAGS=8 LD_PRELOAD=$P138 "$PY" "$S/bitwise2.py" --mode bi_noenv --dtypes float32 --out "$OUT/E_lib138_flags8_fp32.json" "${A[@]}"
