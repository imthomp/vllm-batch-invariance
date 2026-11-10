#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3:00:00
# Better default, selection and held-out kernel check (PROTOCOL.md). Usage: sbatch --export=ALL,PY=...,OUT=... slurm/generic.sbatch
set -eu
S=${SRC:-$(pwd)/src}
TRAIN="12288x2048;2048x6144;4096x2048;151936x2048;2048x2048;24576x4096;4096x12288;6144x4096;4096x4096;151936x4096;28672x4096;4096x14336;128256x4096;5120x2880;2880x4096;201088x2880;32x2880;128x2880;256x6144;160x6144;19360x6144;38720x6144;384x5120;32x5120;128x512;1024x5120;16160x5120;32320x5120"
TEST="4608x3584;3584x3584;37888x3584;3584x18944;152064x3584;8192x3584;3584x4096;28672x3584;3584x14336;256000x3584"
nvidia-smi --query-gpu=name,driver_version --format=csv
"$PY" "$S/tune_generic.py" --train "$TRAIN" --test "$TEST" --out "$OUT/generic.json"
