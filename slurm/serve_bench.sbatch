#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
# Serving benchmark only (src/serve_bench.sh). The tuned arm loads a tune_bi.py output through src/tunedhook.
# Usage: sbatch --export=ALL,PY=...,MODEL=...,VBI_TUNED=<tuned json>,OUT=<dir>[,LIBPATH=<dir for LD_LIBRARY_PATH>] slurm/serve_bench.sbatch
set -u
export OPENSSL_CONF=/dev/null HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 VLLM_USE_FLASHINFER_SAMPLER=0 VBI_TUNED
[ -n "${LIBPATH:-}" ] && export LD_LIBRARY_PATH=$LIBPATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
S=${SRC:-$(pwd)/src}
OVERLAY=$S/tunedhook bash "$S/serve_bench.sh"
grep -h "patched tuned table" "$OUT"/server_tuned_r*.log | head -3
